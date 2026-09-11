from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.config import Settings
from src.reviewer import CATEGORIES, review_category
from src.schemas import Finding


class ReviewState(TypedDict, total=False):
    context: str
    findings: list[Finding]


def make_node(settings: Settings, category: str):
    async def node(state: ReviewState):
        new_findings = await review_category(
            settings=settings,
            category=category,
            context=state["context"],
        )
        return {"findings": state.get("findings", []) + new_findings}

    return node


def build_graph(settings: Settings):
    builder = StateGraph(ReviewState)

    for category in CATEGORIES:
        builder.add_node(category, make_node(settings, category))

    builder.add_edge(START, "security")
    builder.add_edge("security", "standards")
    builder.add_edge("standards", "tests")
    builder.add_edge("tests", "performance")
    builder.add_edge("performance", END)

    return builder.compile()
