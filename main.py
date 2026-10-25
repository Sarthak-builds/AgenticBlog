from __future__ import annotations
import operator
from pydoc import describe
from typing import TypedDict, List, Annotated, Optional
from openai import decisions
from pydantic import BaseModel, Field
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    max_tokens=8192,
)

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-5 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(
        ...,
        description="Target word count for this section (120-450).",
    )
    section_type: Literal[
        "intro", "core", "examples", "checklist", "common_mistakes", "conclusion"
    ] = Field(
        ...,
        description="Use 'common_mistakes' exactly once in the plan.",
    )
    tags: List[str] = Field(
        ...,
        description="Short topical tags for this section.",
    )
    requires_research: bool = Field(
        ...,
        description="Whether this section needs external research/evidence.",
    )
    requires_citations: bool = Field(
        ...,
        description="Whether this section needs citations to sources.",
    )
    requires_code: bool = Field(
        ...,
        description="Whether this section should include a code example.",
    )

class Plan(BaseModel):
    blog_title: str = Field(..., description="Title of the blog post.")
    audience: str = Field(..., description="Who this blog is for.")
    tone: str = Field(..., description="Writing tone (e.g., practical, crisp)")
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = Field(
        ...,
        description="Kind of blog post being planned.",
    )
    constraints: List[str] = Field(
        ...,
        description="Constraints the writers must honor (length, style, requirements).",
    )
    tasks: List[Task]

class RouterDecision(BaseModel):
    needs_research: bool = Field(
        ...,
        description="Whether external research is needed for this topic.",
    )
    mode: Literal["closed_book", "hybrid", "open_book"] = Field(
        ...,
        description="closed_book = no research; hybrid = light research; open_book = heavy research.",
    )
    reason: str = Field(
        ...,
        description="One sentence explaining the decision.",
    )
    queries: List[str] = Field(
        ...,
        description="Search queries to run if research is needed.",
    )

class EvidenceItem(BaseModel):
    title: str = Field(..., description="Title of the source.")
    url: str = Field(..., description="URL of the source.")
    published_at: Optional[str] = Field(
        None, description="Publication date if known (ISO format)."
    )
    snippet: Optional[str] = Field(
        None, description="Short relevant excerpt from the source."
    )
    source: Optional[str] = Field(
        None, description="Publisher/site name, e.g. 'OpenAI Blog'."
    )

class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(
        ...,
        description="Collected evidence items from research.",
    )

class State(TypedDict):
    topic:str
    mode:str
    needs_research:bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    #reducer for results from workers get concatenated automatically
    sections: Annotated[List[tuple[int,str]], operator.add]
    final:str

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3-4 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
If needs_research=true:"""
def router_node(state:State)-> dict:

    topic= state['topic']
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"TOpic: {topic}")
        ]
    )
    return {
          "needs_research": decision.needs_research,
          "mode": decision.mode,
          "queries": decision.queries
    }


def router_next(state:State) -> str:
    return "research" if state['needs_research'] else 'orchestrator'

@traceable(name="orchestrator -> plan")
def orchestrator(state: State) -> dict:

    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content=(
                    "You are a senior technical writer and developer advocate. Your job is to produce "
                    "a highly actionable outline for a technical blog post.\n\n"
                    "Hard requirements:\n"
                    "- Create 5-7 sections (tasks) that fit a technical blog.\n"
                    "- Each section must include:\n"
                    "  1) goal (1 sentence: what the reader can do/understand after the section)\n"
                    "  2) 3-5 bullets that are concrete, specific, and non-overlapping\n"
                    "  3) target word count (120-450)\n"
                    "Include EXACTLY ONE section with section_type='common_mistakes'.\n\n"
                    "Make it technical (not generic):\n"
                    "- Assume the reader is a developer; use correct terminology.\n"
                    "- Prefer design/engineering structure: problem → intuition → approach → implementation → "
                    "trade-offs → testing/observability → conclusion.\n"
                    "- Bullets must be actionable and testable (e.g., 'Show a minimal code snippet for X', "
                    "'Explain why Y fails under Z condition', 'Add a checklist for production readiness').\n"
                    "- Explicitly include at least ONE of the following somewhere in the plan (as bullets):\n"
                    "  * a minimal working example (MWE) or code sketch\n"
                    "  * edge cases / failure modes\n"
                    "  * performance/cost considerations\n"
                    "  * security/privacy considerations (if relevant)\n"
                    "  * debugging tips / observability (logs, metrics, traces)\n"
                    "- Avoid vague bullets like 'Explain X' or 'Discuss Y'. Every bullet should state what "
                    "to build/compare/measure/verify.\n\n"
                    "Ordering guidance:\n"
                    "- Start with a crisp intro and problem framing.\n"
                    "- Build core concepts before advanced details.\n"
                    "- Include one section for common mistakes and how to avoid them.\n"
                    "- End with a conclusion and next steps.\n"
                )
            ),
            HumanMessage(content=f'Topic: {state['topic']}')
        ]
    )
    return {'plan':plan}

# we do not know, how many workers we have, so we use send api for number of workers for each number of tasks in the plan output.

@traceable(name="fanout -> dispatch workers")
def fanout(state:State):
    return [Send('worker', {'task': task, "topic": state["topic"], "plan":state["plan"]})
            for task in state["plan"].tasks]

@traceable(name="worker -> write section")
def worker(payload: dict) -> dict:

    task = payload['task']
    topic = payload['topic']
    plan = payload['plan']

    blog_title = plan.blog_title

    content = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a senior technical writer and developer advocate. Write ONE section of a technical blog post in Markdown.\n"
                    "Hard constraints:\n"
                    "- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).\n"
                    "- Stay close to the Target words (±15%).\n"
                    "- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).\n\n"
                    "Technical quality bar:\n"
                    "- Be precise and implementation-oriented (developers should be able to apply it).\n"
                    "- Prefer concrete details over abstractions: APIs, data structures, protocols, and exact terms.\n"
                    "- When relevant, include at least one of:\n"
                    "  * a small code snippet (minimal, correct, and idiomatic)\n"
                    "  * a tiny example input/output\n"
                    "  * a checklist of steps\n"
                    "  * a diagram described in text (e.g., 'Flow: A -> B -> C')\n"
                    "- Explain trade-offs briefly (performance, cost, complexity, reliability).\n"
                    "- Call out edge cases / failure modes and what to do about them.\n"
                    "- If you mention a best practice, add the 'why' in one sentence.\n\n"
                    "Markdown style:\n"
                    "- Start with a '## <Section Title>' heading.\n"
                    "- Use short paragraphs, bullet lists where helpful, and code fences for code.\n"
                    "- Avoid fluff. Avoid marketing language.\n"
                    "- If you include code, keep it focused on the bullet being addressed.\n"
                )
            ),
            HumanMessage(
                content=                          f'Blog: {plan.blog_title}\n'
                         f'Audience: {plan.audience}\n'
                         f'Tone: {plan.tone}\n'
                         f'Topic: {topic}\n\n'
                         f'Section: {task.title}\n'
                         f'Goal: {task.goal}\n'
                         f'Bullets to cover: {", ".join(task.bullets)}\n'
                         f'Target length: {task.target_words} words\n\n'
                         "return only the section content in markdown."
            ),
        ]
    ).content

    section_md = extract_text(content).strip()

def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
            for part in content
        )
    return str(content)

# reduceeer

from pathlib import Path

@traceable(name="reducer -> assemble blog")
def reducer(state:State)-> dict:
    title = state['plan'].blog_title
    body = "\n\n".join(state["sections"]).strip()

    final_md =f'# {title}\n\n{body}\n'

    filename = title.lower().replace(" ", "-") + ".md"
    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")

    return {'final': final_md}


graph = StateGraph(State)
graph.add_node("orchestrator", orchestrator)
graph.add_node("worker", worker)
graph.add_node("reducer", reducer)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges("orchestrator", fanout, ["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)

app = graph.compile()

out = app.invoke({"topic": "write a blog on self Attention", "sections": []})

print(out['final'])