import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

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
    """Generate test scenarios for the Developer's code."""

    prompt = f"""
You are a Senior QA Engineer.

You are working ONLY as the TESTER.

The Developer has already completed the coding task.
The Developer must NOT participate in testing.

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

def task_input_node(state: CrewState):

    return {
        "next_step": "developer"
    }


# ------------------------------------------
# DEVELOPER
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
# TESTER
# ------------------------------------------

def real_time_tester(state: CrewState):

    task = state["messages"][-1].content

    developer_code = state.get("code", "")

    if not developer_code:
        return {
            "report": "Testing failed: No code was provided by Developer."
        }

    # Tester independently generates test cases
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

    # Tester executes Developer's completed code
    execution_result = run_python_code.invoke({
        "code": developer_code
    })

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
# MANAGER
# ------------------------------------------

def manager_decision_node(state: CrewState):

    return {
        "next_step": "archiver"
    }


# ------------------------------------------
# ARCHIVER
# ------------------------------------------

def archiver_node(state: CrewState):

    return {
        "next_step": "exit"
    }


# ==========================================
# 5. GRAPH CONSTRUCTION
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


# MANAGER -> ARCHIVER

def route_from_decision(state: CrewState):

    return "archiver"


rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)


# ARCHIVER -> END

rt_workflow.add_edge(
    "archiver",
    END
)


# Compile

rt_app = rt_workflow.compile()


# ==========================================
# 6. FASTAPI APPLICATION
# ==========================================

app = FastAPI()


# ==========================================
# WEBSITE HOME PAGE
# ==========================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>LangGraph Developer Tester</title>

        <style>

            body {
                font-family: Arial, sans-serif;
                background: #f4f4f4;
                margin: 0;
                padding: 0;
            }

            .container {
                max-width: 900px;
                margin: 50px auto;
                background: white;
                padding: 35px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }

            h1 {
                text-align: center;
                margin-bottom: 10px;
            }

            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }

            label {
                font-weight: bold;
                display: block;
                margin-bottom: 10px;
            }

            textarea {
                width: 100%;
                height: 140px;
                padding: 12px;
                font-size: 16px;
                border: 1px solid #ccc;
                border-radius: 8px;
                box-sizing: border-box;
                resize: vertical;
            }

            button {
                margin-top: 15px;
                width: 100%;
                padding: 13px;
                font-size: 17px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
                background: #333;
                color: white;
                cursor: pointer;
            }

            button:hover {
                background: #555;
            }

            .flow {
                text-align: center;
                margin: 25px 0;
                font-weight: bold;
                color: #555;
            }

        </style>

    </head>

    <body>

        <div class="container">

            <h1>LangGraph Developer → Tester</h1>

            <p class="subtitle">
                Enter a coding task and let the Developer and Tester agents work on it.
            </p>

            <div class="flow">
                Developer → Tester → Manager
            </div>

            <form action="/run" method="post">

                <label for="task">
                    Enter your coding task:
                </label>

                <textarea
                    id="task"
                    name="task"
                    placeholder="Example: Write a Python program to perform merge sort."
                    required
                ></textarea>

                <button type="submit">
                    Run Workflow
                </button>

            </form>

        </div>

    </body>

    </html>
    """


# ==========================================
# RUN WORKFLOW
# ==========================================

@app.post("/run", response_class=HTMLResponse)
def run_workflow(task: str = Form(...)):

    initial_state: CrewState = {
        "messages": [
            HumanMessage(content=task)
        ],
        "next_step": "developer",
        "code": None,
        "report": None
    }

    result = rt_app.invoke(initial_state)

    generated_code = result.get(
        "code",
        "No code generated."
    )

    test_report = result.get(
        "report",
        "No test report generated."
    )

    # Convert newlines to HTML
    generated_code_html = (
        generated_code
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    test_report_html = (
        test_report
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    return f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>LangGraph Results</title>

        <style>

            body {{
                font-family: Arial, sans-serif;
                background: #f4f4f4;
                margin: 0;
                padding: 30px;
            }}

            .container {{
                max-width: 1000px;
                margin: auto;
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}

            h1 {{
                text-align: center;
            }}

            h2 {{
                margin-top: 30px;
            }}

            pre {{
                background: #f1f1f1;
                padding: 20px;
                border-radius: 8px;
                white-space: pre-wrap;
                overflow-x: auto;
            }}

            .back {{
                display: block;
                text-align: center;
                margin-top: 30px;
                padding: 12px;
                background: #333;
                color: white;
                text-decoration: none;
                border-radius: 8px;
            }}

        </style>

    </head>

    <body>

        <div class="container">

            <h1>Workflow Result</h1>

            <h2>🧑‍💻 Developer - Generated Code</h2>

            <pre>{generated_code_html}</pre>

            <h2>🧪 Tester - Test Report</h2>

            <pre>{test_report_html}</pre>

            <a class="back" href="/">
                ← Try Another Task
            </a>

        </div>

    </body>

    </html>
    """


# ==========================================
# RENDER STARTUP
# ==========================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get("PORT", 10000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
