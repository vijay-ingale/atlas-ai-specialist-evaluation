import re
from src.core.errors import InvalidInputError

def normalize(message: str) -> str:
    if not isinstance(message, str) or not message.strip():
        raise InvalidInputError("Provide a non-empty customer message.")
    if len(message) > 5000:
        raise InvalidInputError("Messages must be at most 5,000 characters.")
    return re.sub(r"\s+", " ", message).strip().lower()
