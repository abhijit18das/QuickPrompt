import pytest
from unittest.mock import patch, MagicMock
from quickprompt import generate_prompt
from quickprompt.models import ModelConfig, SupportedProviders
from quickprompt.exceptions import UnsupportedProviderError

@pytest.fixture
def mock_openai_response():
    return MagicMock(choices=[MagicMock(message=MagicMock(content="Mocked OpenAI response"))])

@pytest.fixture
def mock_anthropic_response():
    return MagicMock(completion="Mocked Anthropic response")

@pytest.fixture
def mock_groq_response():
    return MagicMock(choices=[MagicMock(message=MagicMock(content="Mocked Groq response"))])

def test_generate_prompt_openai(mock_openai_response: MagicMock):
    with patch('openai.ChatCompletion.create', return_value=mock_openai_response):
        model_config = ModelConfig(SupportedProviders.OPENAI, "gpt-4", "fake-api-key")
        prompt = generate_prompt(
            model_config=model_config,
            input_type="Text",
            input_format="JSON",
            use_case="Sentiment Analysis",
            output_format="JSON",
            output_type="Classification"
        )
        assert prompt == "Mocked OpenAI response"

def test_generate_prompt_anthropic(mock_anthropic_response: MagicMock):
    with patch('anthropic.Anthropic') as MockAnthropic:
        MockAnthropic.return_value.completions.create.return_value = mock_anthropic_response
        model_config = ModelConfig(SupportedProviders.ANTHROPIC, "claude-2", "fake-api-key")
        prompt = generate_prompt(
            model_config=model_config,
            input_type="Text",
            input_format="JSON",
            use_case="Sentiment Analysis",
            output_format="JSON",
            output_type="Classification"
        )
        assert prompt == "Mocked Anthropic response"

def test_generate_prompt_groq(mock_groq_response: MagicMock):
    with patch('groq.Client') as MockGroq:
        MockGroq.return_value.chat.completions.create.return_value = mock_groq_response
        model_config = ModelConfig(SupportedProviders.GROQ, "llama3-7b", "fake-api-key")
        prompt = generate_prompt(
            model_config=model_config,
            input_type="Text",
            input_format="JSON",
            use_case="Sentiment Analysis",
            output_format="JSON",
            output_type="Classification"
        )
        assert prompt == "Mocked Groq response"

def test_generate_prompt_unsupported_provider():
    with pytest.raises(UnsupportedProviderError):
        model_config = ModelConfig("unsupported_provider", "model", "fake-api-key")
        generate_prompt(
            model_config=model_config,
            input_type="Text",
            input_format="JSON",
            use_case="Sentiment Analysis",
            output_format="JSON",
            output_type="Classification"
        )

def test_generate_prompt_with_additional_context(mock_openai_response: MagicMock):
    with patch('openai.ChatCompletion.create', return_value=mock_openai_response):
        model_config = ModelConfig(SupportedProviders.OPENAI, "gpt-4", "fake-api-key")
        prompt = generate_prompt(
            model_config=model_config,
            input_type="Text",
            input_format="JSON",
            use_case="Sentiment Analysis",
            output_format="JSON",
            output_type="Classification",
            additional_context={"max_tokens": 100, "temperature": 0.7}
        )
        assert "Mocked OpenAI response" in prompt
        assert "Additional Context:" in prompt
        assert "max_tokens: 100" in prompt
        assert "temperature: 0.7" in prompt