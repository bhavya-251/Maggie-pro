import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# 1. LLM INITIALIZATION
# ==========================================

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)

llm = llm_flash


# ==========================================
# 2. STATE DEFINITION
# ==========================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ==========================================
# 3. TOOLS
# ==========================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return the standard output or error trace."""

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        local_scope = {}

        exec(clean_code, {}, local_scope)

        result = new_stdout.getvalue()

    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"

    finally:
        sys.stdout = old_stdout

    return result.strip() if result.strip() else "Success (no terminal output)"


@tool
def generate_test_cases(
    task_description: str,
    developer_code: str
) -> str:
    """Generate specific test scenarios for the given coding task and developer-generated code."""

    prompt = f"""
You are a Senior QA Engineer.

You are working ONLY as the TESTER.

The Developer has already completed the coding task.
The Developer must NOT participate in testing.

Your job is to test the code provided below.

CODING TASK:
{task_description}

CODE PRODUCED BY THE DEVELOPER:
{developer_code}

Generate 3 to 5 highly specific test scenarios for this exact code.

Include:
1. Standard/normal cases
2. Edge cases
3. Invalid input cases if applicable

Return the test scenarios as a numbered list.

Do NOT rewrite the code.
Do NOT modify the code.
Do NOT ask the Developer to change anything.
Do NOT generate a new solution.
"""

    response = llm_flash.invoke(prompt)

    return response.content if hasattr(response, "content") else str(response)


# ==========================================
# 4. GRAPH NODES
# ==========================================


# ------------------------------------------
# TASK INPUT NODE
# ------------------------------------------

def task_input_node(state: CrewState):

    return {
        "next_step": "developer"
    }


# ------------------------------------------
# DEVELOPER NODE
# ------------------------------------------

def real_time_developer(state: CrewState):

    task = state["messages"][-1].content

    dev_prompt = f"""
Write a clean Python script to solve this coding task:

{task}

Only return the Python code.
No explanation.
No markdown formatting.
Do not include ```python.
"""

    response = llm_flash.invoke(dev_prompt)

    content = response.content

    if isinstance(content, list):

        code_parts = []

        for item in content:
            if isinstance(item, dict):
                code_parts.append(item.get("text", ""))
            else:
                code_parts.append(str(item))

        code_str = "\n".join(code_parts)

    else:
        code_str = str(content)

    code_str = (
        code_str
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    return {
        "code": code_str
    }


# ------------------------------------------
# TESTER NODE
# ------------------------------------------
#
# Developer has already finished.
# Tester receives the Developer's code
# through state["code"].
#
# Developer is NOT called here.
# ------------------------------------------

def real_time_tester(state: CrewState):

    task = state["messages"][-1].content

    developer_code = state.get("code", "")

    if not developer_code:
        return {
            "report": "Testing failed: No code was provided by Developer."
        }

    # Generate test cases independently
    test_cases = generate_test_cases.invoke({
        "task_description": task,
        "developer_code": developer_code
    })

    if isinstance(test_cases, list):

        cases_parts = []

        for item in test_cases:
            if isinstance(item, dict):
                cases_parts.append(item.get("text", ""))
            else:
                cases_parts.append(str(item))

        cases_str = "\n".join(cases_parts)

    else:
        cases_str = str(test_cases)

    # Execute Developer's code
    execution_result = run_python_code.invoke({
        "code": developer_code
    })

    # Create report
    report = f"""
### EXECUTION OUTPUT:

{execution_result}

### TEST SCENARIOS EVALUATED:

{cases_str}
"""

    return {
        "report": report
    }


# ------------------------------------------
# MANAGER NODE
# ------------------------------------------

def manager_decision_node(state: CrewState):

    # In Render there is no input().
    # The API request supplies the command.

    command = state.get("next_step", "store")

    if command == "another":
        return {
            "next_step": "task_input"
        }

    return {
        "next_step": "archiver"
    }


# ------------------------------------------
# ARCHIVER NODE
# ------------------------------------------

def archiver_node(state: CrewState):

    return {
        "next_step": "exit"
    }


# ==========================================
# 5. GRAPH CONSTRUCTION & ROUTING
# ==========================================

rt_workflow = StateGraph(CrewState)


rt_workflow.add_node(
    "task_input",
    task_input_node
)

rt_workflow.add_node(
    "developer",
    real_time_developer
)

rt_workflow.add_node(
    "tester",
    real_time_tester
)

rt_workflow.add_node(
    "manager_decision",
    manager_decision_node
)

rt_workflow.add_node(
    "archiver",
    archiver_node
)


# START -> TASK INPUT

rt_workflow.add_edge(
    START,
    "task_input"
)


# TASK INPUT -> DEVELOPER

def route_from_input(state: CrewState):

    return "developer"


rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input
)


# DEVELOPER -> TESTER

rt_workflow.add_edge(
    "developer",
    "tester"
)


# TESTER -> MANAGER

rt_workflow.add_edge(
    "tester",
    "manager_decision"
)


# MANAGER ROUTING

def route_from_decision(state: CrewState):

    if state.get("next_step") == "archiver":
        return "archiver"

    return "task_input"


rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)


# ARCHIVER -> END

rt_workflow.add_edge(
    "archiver",
    END
)


# Compile workflow

rt_app = rt_workflow.compile()


# ==========================================
# 6. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="LangGraph Developer Tester",
    description="Developer → Tester → Manager LangGraph workflow"
)


# ==========================================
# REQUEST MODEL
# ==========================================

class TaskRequest(BaseModel):
    task: str
    command: str = "store"


# ==========================================
# HOME ROUTE
# ==========================================

@app.get("/")
def home():

    return {
        "message": "LangGraph Developer → Tester workflow is running.",
        "endpoint": "/run",
        "method": "POST"
    }


# ==========================================
# RUN WORKFLOW
# ==========================================

@app.post("/run")
def run_workflow(request: TaskRequest):

    initial_state: CrewState = {
        "messages": [
            HumanMessage(content=request.task)
        ],
        "next_step": request.command,
        "code": None,
        "report": None
    }

    # Run Developer -> Tester -> Manager
    result = rt_app.invoke(initial_state)

    return {
        "task": request.task,
        "generated_code": result.get("code"),
        "test_report": result.get("report"),
        "next_step": result.get("next_step")
    }


# ==========================================
# RENDER STARTUP
# ==========================================

if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )