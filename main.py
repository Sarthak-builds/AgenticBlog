"""AgenticBlog chatbot backend.

A LangGraph chatbot: Gemini-only LLM, tool calling, SQLite-backed conversation
memory (thread based), MCP tool loading (local stdio math server + optional
remote server), and LangSmith tracing wired through environment variables.

Async end-to-end because MCP tools loaded via langchain-mcp-adapters are
async-only; a dedicated background event loop bridges sync callers (Streamlit).
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path
from typing import Annotated, TypedDict

import aiosqlite
import requests
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# Dedicated background event loop so async-only MCP work and async graph runs
# can be driven from sync code safely, also inside Streamlit reruns.
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def run_async(coro, timeout: float | None = None):
    """Run a coroutine on the background loop and block for the result."""
    future = asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)
    return future.result(timeout)


def submit_async_task(coro):
    """Schedule a coroutine on the background loop without blocking."""
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


# ---------------------------------------------------------------------------
# 1. LLM (Gemini only)
# ---------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    max_tokens=8192,
)


# ---------------------------------------------------------------------------
# 2. Built-in tools
# ---------------------------------------------------------------------------
search_tool = TavilySearchResults(max_results=int(os.getenv("TAVILY_MAX_RESULTS", "5")))


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.

    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using the Alpha Vantage API.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {"error": "Set ALPHA_VANTAGE_API_KEY in .env to use this tool"}
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=15)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 3. MCP tools (local stdio math server + optional remote MCP server)
# ---------------------------------------------------------------------------
_LOCAL_MCP = (
    "math",
    {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(Path(__file__).with_name("mcp_server.py"))],
    },
)

_REMOTE_MCP = (
    "extras",
    {
        "transport": "streamable_http",
        "url": os.getenv(
            "MCP_REMOTE_URL", "https://splendid-gold-dingo.fastmcp.app/mcp"
        ),
    },
)


def _load_mcp_tools() -> list[BaseTool]:
    """Load MCP tools server by server so one dead server cannot kill the rest."""
    loaded: list[BaseTool] = []
    for name, config in (_LOCAL_MCP, _REMOTE_MCP):
        try:
            client = MultiServerMCPClient({name: config})
            loaded.extend(run_async(client.get_tools(), timeout=60))
        except Exception:
            continue
    return loaded


def load_mcp_tools() -> list[BaseTool]:
    try:
        return _load_mcp_tools()
    except Exception:
        return []


mcp_tools = load_mcp_tools()

tools = [search_tool, calculator, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm


# ---------------------------------------------------------------------------
# 4. State
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# 5. Nodes
# ---------------------------------------------------------------------------
async def chat_node(state: ChatState):
    """LLM node that may answer directly or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)


# ---------------------------------------------------------------------------
# 6. Checkpointer (SQLite / thread-persistent memory)
# ---------------------------------------------------------------------------
DB_PATH = Path(os.getenv("CHATBOT_DB", "chatbot.db"))


async def _init_checkpointer():
    conn = await aiosqlite.connect(DB_PATH)
    return AsyncSqliteSaver(conn)


checkpointer: AsyncSqliteSaver = run_async(_init_checkpointer())


# ---------------------------------------------------------------------------
# 7. Graph
# ---------------------------------------------------------------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 8. Thread helpers
# ---------------------------------------------------------------------------
async def _alist_threads() -> list[str]:
    all_threads: set[str] = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return sorted(all_threads)


def retrieve_all_threads() -> list[str]:
    return run_async(_alist_threads())


def delete_thread(thread_id: str) -> None:
    run_async(checkpointer.adelete_thread(thread_id))
