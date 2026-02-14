import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]
    rewritten_prompt: str | None
    plan: str | None


CHAT_GPT_DIR = "chat_gpt"


@tool
def create_folder(folder_name: str) -> str:
    """Create a folder inside chat_gpt/ with the given name."""
    os.makedirs(CHAT_GPT_DIR, exist_ok=True)
    safe_name = os.path.basename(folder_name.strip())
    path = os.path.join(CHAT_GPT_DIR, safe_name)
    try:
        os.makedirs(path, exist_ok=True)
        return f"Folder created at {path}"
    except Exception as e:
        return f"Failed to create folder: {e}"


@tool
def create_code_file(filename: str, content: str, folder_name: str | None = None) -> str:
    """Create or overwrite a code file inside the chat_gpt/ (or a subfolder) with the given content.

    Args:
        filename: File name only, e.g. "index.html" (no path separators).
        content: Full file content to write.
        folder_name: Optional subfolder under chat_gpt/, e.g. "netflix_landing".
    """
    base_dir = CHAT_GPT_DIR
    if folder_name:
        base_dir = os.path.join(CHAT_GPT_DIR, os.path.basename(folder_name.strip()))
    os.makedirs(base_dir, exist_ok=True)
    safe_name = os.path.basename(filename)
    path = os.path.join(base_dir, safe_name)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File created/updated at {path}"
    except Exception as e:
        return f"Failed to create/update file: {e}"


@tool
def delete_folder(folder_name: str) -> str:
    """Delete a folder inside chat_gpt/ and all of its contents."""
    import shutil

    base = os.path.join(CHAT_GPT_DIR, os.path.basename(folder_name.strip()))
    if not os.path.exists(base):
        return f"Folder {base} does not exist."
    try:
        shutil.rmtree(base)
        return f"Folder {base} has been deleted."
    except Exception as e:
        return f"Failed to delete folder: {e}"


@tool
def delete_file(filename: str, folder_name: str | None = None) -> str:
    """Delete a file inside chat_gpt/ (or a subfolder)."""
    base_dir = CHAT_GPT_DIR
    if folder_name:
        base_dir = os.path.join(CHAT_GPT_DIR, os.path.basename(folder_name.strip()))
    path = os.path.join(base_dir, os.path.basename(filename))
    if not os.path.exists(path):
        return f"File {path} does not exist."
    try:
        os.remove(path)
        return f"File {path} has been deleted."
    except Exception as e:
        return f"Failed to delete file: {e}"


@tool
def run_project() -> str:
    """Start a simple HTTP server to serve files from the chat_gpt/ folder at http://localhost:8000."""
    import threading
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

    try:
        os.makedirs(CHAT_GPT_DIR, exist_ok=True)
        os.chdir(CHAT_GPT_DIR)
    except Exception as e:
        return f"Failed to prepare project directory: {e}"

    def serve():
        with ThreadingHTTPServer(("localhost", 8000), SimpleHTTPRequestHandler) as httpd:
            httpd.serve_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return "Project server started at http://localhost:8000"


# Main tool-enabled model
llm = init_chat_model(
    model_provider="openai", model="gpt-4.1"
)
llm_with_tool = llm.bind_tools(tools=[create_folder, create_code_file, delete_folder, delete_file, run_project])

# Lightweight helper model for rewrite / planning (no tools)
helper_llm = init_chat_model(
    model_provider="openai", model="gpt-4.1-mini"
)


def rewrite_node(state: State) -> State:
    """Rewrite the latest user request into a precise instruction."""
    user_msg = None
    for m in reversed(state["messages"]):
        if getattr(m, "type", None) == "human" or getattr(m, "role", None) == "user":
            user_msg = m
            break
    if user_msg is None:
        return state

    content = getattr(user_msg, "content", None) or ""
    prompt = (
        "Rewrite the following user request into a single, precise instruction, "
        "keeping all important details but removing ambiguity. "
        "Respond with only the rewritten instruction.\n\n"
        f"User request:\n{content}"
    )
    rewritten = helper_llm.invoke(prompt)
    rewritten_text = getattr(rewritten, "content", "") or ""

    return {**state, "rewritten_prompt": rewritten_text}


def plan_node(state: State) -> State:
    """Produce a short high-level internal plan for how to handle the request."""
    user_text = ""
    for m in reversed(state["messages"]):
        if getattr(m, "type", None) == "human" or getattr(m, "role", None) == "user":
            user_text = getattr(m, "content", "") or ""
            break

    rewritten = state.get("rewritten_prompt") or user_text

    prompt = (
        "You are a silent planner for a coding assistant.\n"
        "Given the original user request and a rewritten, clearer instruction,\n"
        "produce a short, high-level plan (3–6 bullet points) of steps the assistant should take.\n"
        "Focus on understanding, which tools to use (folder/file/delete/run), and output format.\n"
        "Do NOT include code, only the plan. Keep it under 120 words.\n\n"
        f"Original request:\n{user_text}\n\n"
        f"Rewritten instruction:\n{rewritten}"
    )
    plan_msg = helper_llm.invoke(prompt)
    plan_text = getattr(plan_msg, "content", "") or ""

    return {**state, "plan": plan_text}


def chatbot(state: State):
    """Main chat node that uses original text + rewritten prompt + plan with tools."""
    original_user_text = ""
    for m in reversed(state["messages"]):
        if getattr(m, "type", None) == "human" or getattr(m, "role", None) == "user":
            original_user_text = getattr(m, "content", "") or ""
            break

    rewritten = state.get("rewritten_prompt") or original_user_text
    plan = state.get("plan") or ""

    system_prompt = SystemMessage(content=f"""
        You are an AI coding assistant.

        CONTEXT
        -------
        Original user request:
        {original_user_text}

        Rewritten, precise instruction:
        {rewritten}

        High-level internal plan (do not repeat verbatim to the user):
        {plan}

        WORKSPACE RULES
        ----------------
        - All work happens under the `chat_gpt/` directory.
        - For each logical project (e.g. "Netflix landing page"), reuse the SAME folder
          (e.g. `chat_gpt/netflix_landing`) and UPDATE existing files via `create_code_file`
          instead of creating duplicates.

        TOOLS
        -----
        - `create_folder`: create project folders under chat_gpt/.
        - `create_code_file`: create/update files (index.html, style.css, script.js, .py, etc.)
          optionally inside a specific project folder.
        - `delete_folder`: delete a whole project folder under chat_gpt/.
        - `delete_file`: delete specific files under chat_gpt/ or a project folder.
        - `run_project`: serve chat_gpt/ at http://localhost:8000.

        BEHAVIOR
        --------
        - Use tools to actually create/update/delete/run projects, not shell commands.
        - Prefer updating existing project files in-place when the user asks for changes.
        - Keep explanations to the user clear and concise; do not expose the internal plan.
    """)

    message = llm_with_tool.invoke([system_prompt] + state["messages"])
    return {"messages": [message]}


tool_node = ToolNode(tools=[create_folder, create_code_file, delete_folder, delete_file, run_project])

graph_builder = StateGraph(State)

# New pipeline: START -> rewrite -> plan -> chatbot -> tools/END
graph_builder.add_node("rewrite", rewrite_node)
graph_builder.add_node("plan", plan_node)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "rewrite")
graph_builder.add_edge("rewrite", "plan")
graph_builder.add_edge("plan", "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)