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


async def review_all_categories(
    settings: Settings,
    context: str,
) -> list:
    model = build_model(settings)
    structured_model = model.with_structured_output(ReviewResult)

    prompts = []

    for category in CATEGORIES:
        prompts.append(
            build_prompt(category, context)
        )

    combined_prompt = """
You are reviewing a GitHub Pull Request.

Analyze the code carefully for ALL of the following categories:

1. Security
2. Coding standards
3. Tests
4. Performance

Return ALL findings together using the required structured output format.

Here are the review instructions:

""" + "\n\n--- NEXT CATEGORY ---\n\n".join(prompts)

    result = await structured_model.ainvoke(combined_prompt)

    return result.findings