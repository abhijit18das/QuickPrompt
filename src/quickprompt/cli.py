import argparse
import os
from .prompt_generator import generate_prompt
from .models import SupportedProviders, ModelConfig, GroqModels
from .llm_clients import get_available_groq_models

def main():
    parser = argparse.ArgumentParser(description="Quickly generate AI prompts using various LLM providers.")
    parser.add_argument("provider", type=str, choices=[p.value for p in SupportedProviders], help="The LLM provider to use.")
    parser.add_argument("model_name", type=str, help="The specific model name to use.")
    parser.add_argument("input_type", type=str, help="The type of input expected.")
    parser.add_argument("input_format", type=str, help="The format of the input.")
    parser.add_argument("use_case", type=str, help="A description of the use case.")
    parser.add_argument("output_format", type=str, help="The desired format of the output.")
    parser.add_argument("output_type", type=str, help="The type of output expected.")
    parser.add_argument("--additional_context", type=str, nargs="+", help="Additional context in key=value format.")
    
    args = parser.parse_args()
    
    additional_context = {}
    if args.additional_context:
        for item in args.additional_context:
            key, value = item.split("=")
            additional_context[key.strip()] = value.strip()
    
    api_key = os.environ.get(f"{args.provider.upper()}_API_KEY")
    if not api_key:
        raise ValueError(f"API key for {args.provider} not found in environment variables.")

    if args.provider == SupportedProviders.GROQ.value:
        available_models = get_available_groq_models()
        if args.model_name not in available_models:
            print(f"Available Groq models: {', '.join(available_models)}")
            args.model_name = input("Please choose a model from the list above: ")
    
    model_config = ModelConfig(SupportedProviders(args.provider), args.model_name, api_key)
    
    prompt = generate_prompt(
        model_config=model_config,
        input_type=args.input_type,
        input_format=args.input_format,
        use_case=args.use_case,
        output_format=args.output_format,
        output_type=args.output_type,
        additional_context=additional_context
    )
    
    print(prompt)

if __name__ == "__main__":
    main()