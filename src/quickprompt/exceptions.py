class UnsupportedProviderError(Exception):
    """Raised when an unsupported LLM provider is specified."""
    pass

class InvalidInputError(Exception):
    """Raised when invalid input is provided to the prompt generator."""
    pass