from transformers import AutoTokenizer

DEFAULT_MODEL_ID = "openai/gpt-oss-20b"


class RLMTokenizer:
    """
    HuggingFace tokenizer wrapper for RLM.

    Args:
        model_id (str): The HuggingFace model ID or path to use with AutoTokenizer.
            Defaults to "openai/gpt-oss-20b".

    Attributes:
        tokenizer: An instance of HuggingFace's AutoTokenizer.

    Example:
        >>> tok = MythosTokenizer()
        >>> ids = tok.encode("Hello world")
        >>> s = tok.decode(ids)
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
