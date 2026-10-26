"""AgenticBlog chatbot frontend (Streamlit).

Chat UI with persistent threads (SQLite-backed memory), new-chat switching,
thread deletion, live tool-call status, and token streaming from the async
LangGraph backend in main.py (MCP tools make the graph async-only).
"""

import queue
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from main import chatbot, delete_thread, retrieve_all_threads, submit_async_task  # noqa: E402

import streamlit as st  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

st.set_page_config(page_title="AgenticBlog Chat", page_icon="✍️", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def generate_thread_id() -> str:
    return f"chat-{time.strftime('%Y-%m-%d-%H-%M-%S')}"


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def serialize_content(content) -> str:
    """Extract visible text; skip Gemini thinking parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("thinking"):
                    continue
                parts.append(part.get("text", ""))
            else:
                parts.append(getattr(part, "text", ""))
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("✍️ AgenticBlog Chat")
    st.caption("Gemini • tools • SQLite memory • MCP • LangSmith")

    if st.button("➕ New Chat"):
        reset_chat()

    st.subheader("My Chats")
    for thread_id in st.session_state["chat_threads"][::-1]:
        col = st.columns([4, 1])
        if col[0].button(str(thread_id), key=f"sel-{thread_id}", use_container_width=True):
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)
            st.session_state["message_history"] = [
                {
                    "role": "user" if isinstance(m, HumanMessage) else "assistant",
                    "content": serialize_content(m.content),
                }
                for m in messages
                if not isinstance(m, ToolMessage) and serialize_content(m.content)
            ]
        if col[1].button("🗑️", key=f"del-{thread_id}"):
            delete_thread(thread_id)
            st.session_state["chat_threads"].remove(thread_id)
            if st.session_state["thread_id"] == thread_id:
                reset_chat()

st.title("AgenticBlog Agent")
st.divider()

# ---------------------------------------------------------------------------
# Render history
# ---------------------------------------------------------------------------
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type here…")

if user_input:
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    config = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, _metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=config,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, _metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, payload = item
                if message_chunk == "error":
                    raise payload

                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …", state="running"
                        )

                if isinstance(message_chunk, AIMessage):
                    text = serialize_content(message_chunk.content)
                    if text:
                        yield text

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
