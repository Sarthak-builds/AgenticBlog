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
