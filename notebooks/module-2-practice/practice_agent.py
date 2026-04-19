"""
Practice agent (same style as notebooks/module-1): create_agent, @tool, HumanMessage.

Using # %% regions in VS Code / Cursor:
  - Open this file, pick your project interpreter (.venv).
  - Click "Run Cell" on the code lens above each # %% block (or Shift+Enter).
  - Order: run env → tools → build agent once per interactive session.
  - Re-run only the "invoke" blocks to try new messages; the `agent` object stays in memory.

Thread memory (like 1.5_personal_chef): same `thread_id` in config keeps conversation
state in the checkpointer until you restart the Python process (InMemorySaver).
For persistence across process restarts, swap in SqliteSaver from langgraph.checkpoint.sqlite.
"""

from pathlib import Path

# %% Load environment (run once per session)
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")

# %% Tools — same pattern as module-1/1.2_tools.ipynb
from langchain.tools import tool


@tool("square_root")
def square_root(x: float) -> float:
    """Calculate the square root of a number."""
    return x**0.5


@tool
def square(x: float) -> float:
    """Return the number multiplied by itself (x squared)."""
    return x * x


# %% Build agent — run once; expensive (model + graph setup)
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-5-nano",
    tools=[square_root, square],
    system_prompt=(
        "You are an arithmetic wizard. Use your tools to calculate square roots and squares. "
        "Round final answers to 3 decimal places when you report them."
    ),
    checkpointer=InMemorySaver(),
)

THREAD_ID = "practice-thread-1"
config = {"configurable": {"thread_id": THREAD_ID}}

# %% First message — uses tools + returns assistant text
from langchain.messages import HumanMessage

response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="What is the square root of 467? Also what is 12 squared?")
        ]
    },
    config,
)
print(response["messages"][-1].content)

# %% Follow-up in the same thread — prior turns stay in checkpoint; no need to rebuild `agent`
follow_up = agent.invoke(
    {"messages": [HumanMessage(content="Round those two answers to 2 decimals only.")]},
    config,
)
print(follow_up["messages"][-1].content)

# %% Inspect full message list (optional)
from pprint import pprint

pprint(follow_up["messages"])
