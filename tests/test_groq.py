import os
import sys
from typing import Any
import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from quickprompt import generate_prompt, ModelConfig, SupportedProviders, GroqModels
from quickprompt.llm_clients import get_available_groq_models

print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("GROQ_API_KEY set:", "Yes" if os.environ.get("GROQ_API_KEY") else "No")

@pytest.fixture(scope="module")
def groq_api_key():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not found in environment variables")
    return api_key

@pytest.fixture(scope="module")
def available_models(groq_api_key: str):
    try:
        models = get_available_groq_models()
        print(f"Available Groq models: {', '.join(models)}")
        return models
    except Exception as e:
        pytest.skip(f"Failed to fetch available Groq models: {str(e)}")

def test_available_groq_models(available_models: list):
    assert len(available_models) > 0, "No Groq models available"

@pytest.mark.parametrize("model", [model for model in GroqModels])
def test_groq_prompt_generation(groq_api_key: str, available_models: list, model: Any):
    if model.value not in available_models:
        pytest.skip(f"Model {model.value} not available")

    model_config = ModelConfig(
        SupportedProviders.GROQ,
        model.value,
        groq_api_key
    )

    try:
        prompt = generate_prompt(
            model_config=model_config,
            input_type="Text",
            input_format="JSON",
            use_case="Sentiment Analysis",
            output_format="JSON",
            output_type="Classification",
            additional_context={"max_tokens": 100, "temperature": 0.7}
        )

        print(f"\nGenerated Prompt for {model.value}:")
        print(prompt)

        assert len(prompt) > 0, "Generated prompt is empty"
        if model.value == GroqModels.LLAMA_GUARD.value:
            assert "placeholder prompt for Llama Guard" in prompt, "Llama Guard placeholder not found"
        else:
            assert "Sentiment Analysis" in prompt, "Use case not reflected in the prompt"
        assert "JSON" in prompt, "Input/Output format not reflected in the prompt"
    except Exception as e:
        print(f"Error generating prompt for {model.value}: {str(e)}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Failed to generate prompt: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])