# Cursor Vibe Talker

Voice-driven coding assistant built with LangChain, LangGraph, and OpenAI. It listens to your speech, understands your intent, plans the steps, and then uses tools to create, update, delete, and run projects in a controlled workspace.

## Core Architecture

- **Entry point**: `app/main.py`
  - Loads environment variables from `.env`.
  - Connects to MongoDB via `MongoDBSaver` for LangGraph checkpointing.
  - Uses `speech_recognition` + `PyAudio` to continuously capture microphone input.
  - Converts speech to text with `recognize_google` and sends user text into the LangGraph.
  - Streams LangGraph events and prints all messages (human/AI/tool) to the console.
  - Uses OpenAI TTS (`gpt-4o-mini-tts`) to speak back a short summary of each AI response.

- **Graph / Orchestration**: `app/graph.py`
  - Uses **LangGraph** `StateGraph` with a custom `State` type:
    - `messages`: conversation history (LangGraph message state with `add_messages`).
    - `rewritten_prompt`: helper field with a clarified user instruction.
    - `plan`: helper field with a short internal plan / chain of thought.
  - The graph has four main nodes:
    1. `rewrite` – rewrites the last user message into a precise instruction.
    2. `plan` – generates a short internal plan for how to handle the request.
    3. `chatbot` – main tool-using LLM node that uses original text + rewritten prompt + plan.
    4. `tools` – `ToolNode` that executes the registered tools.
  - Edges:
    - `START -> rewrite -> plan -> chatbot`.
    - From `chatbot` a conditional edge via `tools_condition` to `tools` if tools are requested.
    - `tools -> chatbot` to inject tool outputs back into the conversation.
    - `chatbot -> END` when no more tool calls are needed.

- **Models**:
  - `llm` (main): `gpt-4.1` via `init_chat_model`, bound with tools for actual actions.
  - `helper_llm`: `gpt-4.1-mini` without tools, used only for rewrite and planning.

## Chain-of-Thought and Prompt Rewriting

To get more precise and robust behavior, the system performs two internal steps before using tools:

1. **Rewrite Node (`rewrite_node`)**
   - Finds the latest user message.
   - Asks `helper_llm` to rewrite the request into a single, unambiguous instruction.
   - Stores the result in `state.rewritten_prompt`.

2. **Plan Node (`plan_node`)**
   - Reads the original request and the rewritten instruction.
   - Asks `helper_llm` to produce a short high-level plan (3–6 bullet points): which tools to use, what files/folders to touch, and general steps.
   - Stores the plan in `state.plan`.

The **main chatbot node** then uses:

- Original user request
- Rewritten instruction
- Internal plan summary

inside a rich `SystemMessage` that also encodes all workspace rules and available tools.

The internal plan is **not** shown directly to the user; it only guides the final answer and tool calls.

## Tooling & Workspace Management

All tools operate under a controlled root folder:

- `CHAT_GPT_DIR = "chat_gpt"`

Tools defined in `graph.py`:

### 1. `create_folder(folder_name: str)`

- Creates a subfolder inside `chat_gpt/`.
- Normalizes the folder name (`os.path.basename`, `strip()`).
- Ensures `chat_gpt/` exists.
- Returns a message indicating the full path, or an error.

### 2. `create_code_file(filename: str, content: str, folder_name: str | None = None)`

- Creates or **overwrites** a file with the given content.
- Operates either directly under `chat_gpt/` or inside `chat_gpt/<folder_name>/`.
- Used both for first-time project creation and for **updates**:
  - Example: `chat_gpt/netflix_landing/index.html` / `style.css` / `script.js`.
- Returns a confirmation with the exact path written.

### 3. `delete_folder(folder_name: str)`

- Deletes a project folder under `chat_gpt/` and all its contents using `shutil.rmtree`.
- Used when you say things like "delete the previous project" or similar.

### 4. `delete_file(filename: str, folder_name: str | None = None)`

- Deletes a single file under `chat_gpt/` or `chat_gpt/<folder_name>/`.
- Returns a message if the file is missing or successfully deleted.

### 5. `run_project()`

- Starts a simple HTTP server serving the `chat_gpt/` directory at:
  - `http://localhost:8000`
- Uses `ThreadingHTTPServer` in a **daemon thread** so the voice loop continues.

These tools are bound to the main model with:

```python
llm_with_tool = llm.bind_tools(
    tools=[create_folder, create_code_file, delete_folder, delete_file, run_project]
)
```

and exposed to LangGraph via:

```python
tool_node = ToolNode(tools=[create_folder, create_code_file, delete_folder, delete_file, run_project])
```

## Workspace Rules & Behaviors

The system prompt for the `chatbot` node enforces several rules:

- **Workspace root**: all generated code and projects live under `chat_gpt/`.
- **Project folders**: each logical project gets a stable folder name (e.g. `netflix_landing`, `student_study_analysis`).
- **Update, don’t duplicate**:
  - When the user asks for modifications to an existing project, the assistant should update the existing files via `create_code_file` instead of creating a new project/folder.
- **Deletion**:
  - For cleaning up, use `delete_folder` / `delete_file` instead of asking the user to delete manually.
- **Running projects**:
  - For “run this / run the project” type requests, the assistant calls `run_project` and instructs the user to open `http://localhost:8000`.

## Voice Loop & TTS Behavior

In `main.py`:

- The app runs an infinite loop:
  1. Listens from the microphone (`sr.Microphone`, `Recognizer.listen`).
  2. Uses Google Speech Recognition (`recognize_google`) to convert audio → text.
  3. Handles errors gracefully:
     - `UnknownValueError` → asks you to repeat.
     - `RequestError` → prints an error and continues.
  4. Sends the recognized text into `graph.stream(...)` and prints all message events.
- After each full LangGraph turn, it:
  - Extracts the last AI message content.
  - Truncates it to a safe length (~400 chars) to avoid TTS token limit issues.
  - Calls `speak(summary_text)` using `AsyncOpenAI().audio.speech` and `LocalAudioPlayer`.

This gives you:

- Hands-free voice input.
- Live streaming of the LLM + tools behavior in the terminal.
- Spoken summaries of each response.

## Features Summary

- Voice-controlled coding assistant.
- Multi-step reasoning pipeline (rewrite + plan + final answer).
- Safe, structured file/folder management under `chat_gpt/`.
- Ability to:
  - Create projects (HTML/CSS/JS, Python scripts, etc.).
  - Update existing project files in-place.
  - Delete specific files or entire project folders.
  - Run static projects via a local HTTP server.
- MongoDB-backed checkpointing for LangGraph via `MongoDBSaver`.
- Spoken summaries of actions and responses using OpenAI TTS.
