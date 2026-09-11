from langchain_openai import ChatOpenAI

from src.config import Settings
from src.prompts import build_prompt
from src.schemas import ReviewResult


CATEGORIES = ["security", "standards", "tests", "performance"]


def build_model(settings: Settings) -> ChatOpenAI:
    kwargs = {
        "model": settings.openai_model,
        "temperature": 0,
        "api_key": settings.openai_api_key,
    }

    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url

    return ChatOpenAI(**kwargs)


async def review_category(
    settings: Settings,
    category: str,
    context: str,
) -> list:
    model = build_model(settings)
    structured_model = model.with_structured_output(ReviewResult)

    result = await structured_model.ainvoke(
        build_prompt(category, context)
    )
    return result.findings
