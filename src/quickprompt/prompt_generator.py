from typing import Dict, Any
from .models import ModelConfig
from .agent import PromptAgent

def generate_prompt(
    model_config: ModelConfig,
    input_type: str,
    input_format: str,
    use_case: str,
    output_format: str,
    output_type: str,
    additional_context: Dict[str, Any] = None
) -> str:
    agent = PromptAgent(model_config)
    system_message = f"""
    You are an AI assistant specialized in {use_case}.
    Generate a comprehensive system prompt for the following task:

    Use Case: {use_case}
    Input Type: {input_type}
    Input Format: {input_format}
    Output Format: {output_format}
    Output Type: {output_type}

    Ensure that the generated prompt clearly mentions the use case '{use_case}'.
    """

    user_message = "Please generate a high-quality system prompt based on the given parameters."

    prompt = agent.generate_prompt(system_message, user_message)

    if additional_context:
        context_str = "\n".join(f"{key}: {value}" for key, value in additional_context.items())
        prompt += f"\n\nAdditional Context:\n{context_str}"

    return prompt.strip()