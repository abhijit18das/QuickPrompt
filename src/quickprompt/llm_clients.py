from abc import ABC, abstractmethod
from .models import ModelConfig, GroqModels
import os
from groq import Groq

class LLMClient(ABC):
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    @abstractmethod
    def generate_completion(self, system_message: str, user_message: str) -> str:
        pass

class OpenAIClient(LLMClient):
    pass

class AnthropicClient(LLMClient):
    pass

class GroqClient(LLMClient):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.client = Groq(api_key=model_config.api_key)

    def generate_completion(self, system_message: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content

def get_available_groq_models():
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    models = client.models.list()
    print("Raw models data:", models)  # Debug print
    
    available_models = []
    for model in models.data:
        if model.id in [m.value for m in GroqModels]:
            available_models.append(model.id)
    
    print(f"Filtered available models: {available_models}")  # Debug print
    return available_models