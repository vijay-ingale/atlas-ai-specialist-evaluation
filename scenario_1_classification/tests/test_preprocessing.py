import pytest
from src.core.errors import InvalidInputError
from src.services.preprocessing import normalize

def test_preserves_negation_and_error_code():
    assert normalize("  I CANNOT log in\n Error 503  ") == "i cannot log in error 503"

@pytest.mark.parametrize("message", ["", "   ", "a" * 5001])
def test_rejects_invalid_messages(message):
    with pytest.raises(InvalidInputError):
        normalize(message)
