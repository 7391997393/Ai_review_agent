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

    max_retries = 4

    for attempt in range(max_retries):
        try:
            result = await structured_model.ainvoke(
                build_prompt(category, context)
            )
            return result.findings

        except Exception as e:
            error_text = str(e)

            # Gemini free-tier rate limit
            if "429" in error_text or "quota" in error_text.lower():
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)

                    print(
                        f"Gemini rate limit reached for '{category}'. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    await asyncio.sleep(wait_time)
                    continue

            raise

    return []