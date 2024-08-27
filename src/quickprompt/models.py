from enum import Enum

class SupportedProviders(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"

class GroqModels(Enum):
    LLAMA3_70B = "llama3-70b-8192"
    LLAMA3_8B = "llama3-8b-8192"
    # LLAMA3_70B_TOOL_USE = "llama3-groq-70b-8192-tool-use-preview" (Not working)
    LLAMA3_8B_TOOL_USE = "llama3-groq-8b-8192-tool-use-preview"
    LLAMA31_70B = "llama-3.1-70b-versatile"
    LLAMA31_8B = "llama-3.1-8b-instant"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    LLAMA_GUARD = "llama-guard-3-8b"

class ModelConfig:
    def __init__(self, provider: SupportedProviders, model_name: str, api_key: str, **kwargs):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.additional_params = kwargs