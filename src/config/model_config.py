"""Configuration for entity extraction using Pydantic Settings."""

import os
from typing import Literal, Optional
from dotenv import load_dotenv

from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class EntityExtractorConfig(BaseSettings):
    """Configuration for LLM-based entity extraction.
    
    All settings can be overridden via environment variables with prefix ENTITY_EXTRACTOR_
    
    Example .env file:
        ENTITY_EXTRACTOR_PROVIDER=anthropic
        ENTITY_EXTRACTOR_MODEL=claude-haiku-3.5-20241022
        ENTITY_EXTRACTOR_ANTHROPIC_API_KEY=sk-ant-...
        ENTITY_EXTRACTOR_TEMPERATURE=0.0
    """

    # Provider selection
    provider: Literal["anthropic", "openai", "openai_async", "google"] = "anthropic"
    
    # Model configuration
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 4096
    
    # API Keys (read from environment)
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    google_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    
    model_config = SettingsConfigDict(
        env_prefix="ENTITY_EXTRACTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    def get_api_key(self) -> str:
        """Get the appropriate API key based on provider."""
        if self.provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
            return self.anthropic_api_key
        elif self.provider == "openai" or self.provider == "openai_async":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            return self.openai_api_key
        elif self.provider == "google":
            if not self.google_api_key:
                raise ValueError("GOOGLE_API_KEY not set in environment")
            return self.google_api_key
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


# Model recommendations by use case
MODEL_PRESETS = {
    "development": {
        "anthropic": "claude-haiku-3.5-20241022",  # Fast, cheap for testing
        "openai": "gpt-4o-mini",
        "google": "gemini-1.5-flash",
    },
    "production": {
        "anthropic": "claude-sonnet-4-20250514",  # Best accuracy
        "openai": "gpt-4o",
        "google": "gemini-1.5-pro",
    },
    "budget": {
        "anthropic": "claude-haiku-3.5-20241022",  # $1/$5 per 1M tokens
        "openai": "gpt-4o-mini",  # $0.15/$0.60 per 1M tokens,
        "openai_async": "gpt-4o-mini",
        "google": "gemini-1.5-flash",  # $0.075/$0.30 per 1M tokens
    },
}


def get_preset_config(
    preset: Literal["development", "production", "budget"] = "development",
    provider: Literal["anthropic", "openai", "openai_async", "google"] = "anthropic",
) -> EntityExtractorConfig:
    """Get a preset configuration for common use cases.
    
    Args:
        preset: Use case preset (development/production/budget)
        provider: LLM provider
        
    Returns:
        Configured EntityExtractorConfig
    """
    model = MODEL_PRESETS[preset][provider]
    return EntityExtractorConfig(provider=provider, model=model)

processed_file_path = 'data/processed/'
save_path = 'data/kg/processed'


# Example usage
if __name__ == "__main__":
    # Load from environment
    config = EntityExtractorConfig()
    print(f"Provider: {config.provider}")
    print(f"Model: {config.model}")
    print(f"Temperature: {config.temperature}")
    
    # Use preset
    dev_config = get_preset_config("development", "anthropic")
    print(f"\nDevelopment preset: {dev_config.model}")
    
    budget_config = get_preset_config("budget", "openai")
    print(f"Budget preset (OpenAI): {budget_config.model}")
