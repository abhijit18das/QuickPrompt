# QuickPrompt

QuickPrompt is a powerful tool for rapidly generating high-quality prompts for various AI models, tailored to specific use cases and requirements.

## Installation

You can install QuickPrompt using pip:

```bash
pip install quickprompt
```

## Usage

### As a Python Package

```python
from quickprompt import generate_prompt, ModelConfig, SupportedProviders, GroqModels
import os

model_config = ModelConfig(
    SupportedProviders.GROQ,
    GroqModels.LLAMA3_70B.value,
    os.environ.get("GROQ_API_KEY")
)

prompt = generate_prompt(
    model_config=model_config,
    input_type="Text",
    input_format="JSON",
    use_case="Sentiment Analysis",
    output_format="JSON",
    output_type="Classification",
    additional_context={"max_tokens": 100, "temperature": 0.7}
)

print(prompt)
```

### Command Line Interface

```bash
export GROQ_API_KEY=your_groq_api_key
quickprompt groq llama3-70b-8192 "Text" "JSON" "Sentiment Analysis" "JSON" "Classification" --additional_context max_tokens=100 temperature=0.7
```

## Supported Providers and Models

- OpenAI
- Anthropic
- Groq
  - Llama 3 70B
  - Llama 3 8B
  - Llama 3 70B Tool Use (Preview)
  - Llama 3 8B Tool Use (Preview)
  - Llama 3.1 70B (Preview)
  - Llama 3.1 8B (Preview)
  - Mixtral 8x7B

## Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/quickprompt.git
   cd quickprompt
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

5. Set up your API keys:
   ```bash
   export GROQ_API_KEY=your_groq_api_key
   export OPENAI_API_KEY=your_openai_api_key
   export ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

6. Run tests:
   ```bash
   pytest tests/
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.