import asyncio

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

    prompt = build_prompt(category, context)

    # Retry temporary Gemini 429/503 errors.
    max_attempts = 4

    for attempt in range(max_attempts):
        try:
            result = await structured_model.ainvoke(prompt)
            return result.findings

        except Exception as exc:
            error_text = str(exc)

            temporary_error = (
                "429" in error_text
                or "503" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "UNAVAILABLE" in error_text
                or "high demand" in error_text
            )

            if not temporary_error or attempt == max_attempts - 1:
                raise

            wait_seconds = 10 * (attempt + 1)
            print(
                f"Temporary Gemini error. "
                f"Retrying in {wait_seconds} seconds..."
            )

            await asyncio.sleep(wait_seconds)

    return []