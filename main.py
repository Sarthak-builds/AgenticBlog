from __future__ import annotations
import operator
from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    max_tokens=1500,
)

class Task(BaseModel):
    id : int
    title: str
    brief: str = Field(..., description="What to cover")

class Plan(BaseModel):
    blog_title:str
    tasks: List[Task]    

class State(TypedDict):
    topic:str
    plan: Plan
    #reducer for results from workers get concatenated automatically
    sections: Annotated[List[str], operator.add]
    final:str

def orchestrator(state: State) -> dict:

    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content=(
                    "Create a blog plan with 5-7 sections on the following topic."
                )
            ), 
            HumanMessage(content=f'Topic: {state['topic']}')
        ]
    )
    return {'plan':plan}

# we do not know, how many workers we have, so we use send api for number of workers for each number of tasks in the plan output.

def fanout(state:State):
    return [Send('worker', {'task': task, "topic": state["topic"], "plan":state["plan"]})
            for task in state["plan"].tasks]

def worker(payload: dict) -> dict:

    task = payload['task']
    topic = payload['topic']
    plan = payload['plan']

    blog_title = plan.blog_title

    section_md = llm.invoke(
        [
            SystemMessage(content="Write one clear Markdown Section."),
            HumanMessage(
                content= f'Blog: {blog_title}\n'
                         f'"Topic: {topic}\n\n'
                         f'Section: {task.title}\n'
                         f'Brief: {task.brief}\n\n'
                         "return only the section content in markdown."
            ),
        ]
    ).content.strip()

    return {'sections':[section_md]}

# reduceeer

from pathlib import Path

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
