from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.config import Settings
from src.reviewer import review_all_categories
from src.schemas import Finding


class ReviewState(TypedDict, total=False):
    context: str
    findings: list[Finding]


async def review_node(state: ReviewState, settings: Settings):
    findings = await review_all_categories(
        settings=settings,
        context=state["context"],
    )

    return {"findings": findings}


def build_graph(settings: Settings):
    builder = StateGraph(ReviewState)

    builder.add_node(
        "review",
        lambda state: review_node(state, settings),
    )

    builder.add_edge(START, "review")
    builder.add_edge("review", END)

    return builder.compile()