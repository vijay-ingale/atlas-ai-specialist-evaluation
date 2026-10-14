from unittest.mock import Mock
from src.schemas.contracts import Claim, Evidence, GroundedAnswer
from src.services.chat import ChatService
from src.services.guardrails import Guardrails

def test_missing_evidence_never_calls_provider():
    generator = Mock()
    result = ChatService(Mock(retrieve=Mock(return_value=[])), generator, Guardrails()).answer("Where is the service?")
    assert result.status == "insufficient_evidence"
    generator.generate.assert_not_called()

def test_fabricated_citation_is_withheld(passage):
    answer = GroundedAnswer(status="answered", claims=[Claim(text="Restart all services.",
        evidence=[Evidence(source_id="missing", quote=passage.text)])])
    result = ChatService(Mock(retrieve=Mock(return_value=[passage])),
        Mock(generate=Mock(return_value=answer)), Guardrails()).answer("When do we roll back?")
    assert result.status == "insufficient_evidence"
    assert result.claims == []

def test_supported_claim_returns_used_source(passage):
    answer = GroundedAnswer(status="answered", claims=[Claim(text="Rollback uses the documented 2% threshold.",
        evidence=[Evidence(source_id=passage.id, quote=passage.text)])])
    result = ChatService(Mock(retrieve=Mock(return_value=[passage])),
        Mock(generate=Mock(return_value=answer)), Guardrails()).answer("When do we roll back?")
    assert result.status == "answered"
    assert result.sources == [passage]

def test_model_refusal_or_missing_parsed_response_abstains(passage):
    result = ChatService(Mock(retrieve=Mock(return_value=[passage])),
        Mock(generate=Mock(return_value=None)), Guardrails()).answer("When do we roll back?")
    assert result.status == "insufficient_evidence"
