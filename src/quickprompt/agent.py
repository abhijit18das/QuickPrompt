from .models import SupportedProviders, ModelConfig, GroqModels
from .exceptions import UnsupportedProviderError
from .llm_clients import OpenAIClient, AnthropicClient, GroqClient

class PromptAgent:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.client = self._get_client()

    def _get_client(self):
        if self.model_config.provider == SupportedProviders.OPENAI:
            return OpenAIClient(self.model_config)
        elif self.model_config.provider == SupportedProviders.ANTHROPIC:
            return AnthropicClient(self.model_config)
        elif self.model_config.provider == SupportedProviders.GROQ:
            return GroqClient(self.model_config)
        else:
            raise UnsupportedProviderError(f"Unsupported provider: {self.model_config.provider}")

    def generate_prompt(self, system_message: str, user_message: str) -> str:
        if self.model_config.provider == SupportedProviders.GROQ and self.model_config.model_name == GroqModels.LLAMA_GUARD.value:
            return f"This is a placeholder prompt for Llama Guard. Actual prompt generation is not supported for this model.\n\n{system_message}\n{user_message}"
        return self.client.generate_completion(system_message, user_message)