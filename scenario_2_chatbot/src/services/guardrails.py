import re
from src.core.errors import InvalidInputError

SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
INJECTION_PATTERN = re.compile(r"ignore (?:all |previous |the )*instructions|reveal (?:the |your )?(?:system prompt|api key)|bypass (?:the )?guardrails", re.I)

class Guardrails:
    def __init__(self, api_key=""):
        self.api_key = api_key if api_key != "replace_with_your_api_key" else ""

    def contains_secret(self, text):
        return bool(SECRET_PATTERN.search(text) or (self.api_key and self.api_key in text))

    def validate_question(self, question):
        question = question.strip()
        if not question or len(question) > 2000:
            raise InvalidInputError("Question must contain 1–2,000 characters.")
        if self.contains_secret(question):
            raise InvalidInputError("Remove credentials from the question.")
        if INJECTION_PATTERN.search(question):
            raise InvalidInputError("Instruction overrides and secret disclosure are unsupported.")
        return question

    def validate_documents(self, documents):
        if any(self.contains_secret(document.content) for document in documents):
            raise InvalidInputError("A document contains potential credentials; review the corpus.")

    def validate_passages(self, passages):
        if any(self.contains_secret(passage.text) for passage in passages):
            raise InvalidInputError("Retrieved content contains potential credentials; review the corpus.")

    def valid_answer(self, answer, passages):
        if answer is None or answer.status != "answered" or not 1 <= len(answer.claims) <= 12:
            return False
        allowed = {passage.id: passage for passage in passages}
        for claim in answer.claims:
            if not claim.text.strip() or len(claim.text) > 2000 or not claim.evidence or self.contains_secret(claim.text):
                return False
            for evidence in claim.evidence:
                source = allowed.get(evidence.source_id)
                if source is None or len(evidence.quote.strip()) < 12:
                    return False
                if evidence.quote not in source.text or self.contains_secret(evidence.quote):
                    return False
        return True
