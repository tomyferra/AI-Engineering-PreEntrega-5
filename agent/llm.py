"""Factory del chat model según LLM_PROVIDER (openai | anthropic | gemini | openrouter)."""
from langchain_core.language_models import BaseChatModel

from config import OPENROUTER_BASE_URL, LLMProvider, LLMSettings


def build_chat_model(settings: LLMSettings) -> BaseChatModel:
    api_key = settings.api_key.get_secret_value()
    if settings.provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.model, api_key=api_key, temperature=settings.temperature)
    if settings.provider == LLMProvider.OPENROUTER:
        # OpenRouter expone una API compatible con OpenAI: mismo cliente, otra base_url.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.model, api_key=api_key, base_url=OPENROUTER_BASE_URL,
                          temperature=settings.temperature)
    if settings.provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=settings.model, api_key=api_key, temperature=settings.temperature)
    if settings.provider == LLMProvider.GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=settings.model, google_api_key=api_key, temperature=settings.temperature)
    raise ValueError(f"LLM provider no soportado: {settings.provider}")
