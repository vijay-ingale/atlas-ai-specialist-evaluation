from src.schemas.contracts import ChatResponse

class ChatService:
    def __init__(self, retrieval, generator, guardrails):
        self.retrieval, self.generator, self.guardrails = retrieval, generator, guardrails

    @staticmethod
    def abstain(message):
        return ChatResponse(status="insufficient_evidence", message=message)

    def answer(self, question):
        question = self.guardrails.validate_question(question)
        passages = self.retrieval.retrieve(question)
        if not passages:
            return self.abstain("The documents do not provide enough relevant evidence.")
        self.guardrails.validate_passages(passages)
        answer = self.generator.generate(question, passages)
        if answer is not None and answer.status == "out_of_scope":
            return ChatResponse(status="out_of_scope", message="Ask about the engineering documents.")
        if not self.guardrails.valid_answer(answer, passages):
            return self.abstain("A supported answer could not be validated. Please clarify your question.")
        used_ids = {e.source_id for claim in answer.claims for e in claim.evidence}
        return ChatResponse(status="answered", claims=answer.claims,
                            sources=[passage for passage in passages if passage.id in used_ids])
