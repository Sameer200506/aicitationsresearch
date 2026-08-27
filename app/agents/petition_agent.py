from ..llm import GROUNDING_RULES
from .base import Agent


class PetitionExtractionAgent(Agent):
    name = "petition_extraction_agent"
    ctx_keys = ["petition_text"]
    system_prompt = f"""You are the Petition Analyzer for an Indian legal research platform.
Extract structured information from the uploaded petition/application/legal notice text below.
Use ONLY facts present in the document; if a field is absent, return an empty list for it.

{GROUNDING_RULES}

Respond ONLY with JSON:
{{"document_type": "writ petition | civil petition | application | legal notice | other",
"parties": [{{"name": "...", "role": "petitioner/respondent/..."}}],
"facts": ["chronological fact statements"],
"dates_and_events": [{{"date": "...", "event": "..."}}],
"authorities_challenged": ["..."],
"legal_issues": [{{"number": 1, "issue": "...", "area": "..."}}],
"reliefs_sought": ["prayer statements"],
"existing_citations": ["citations exactly as they appear in the document"],
"statutory_references": [{{"provision": "Article 226 / Section X of Act Y", "context": "..."}}],
"court": "the court/authority where filed, if identifiable"}}"""


class MissingAuthorityDetector(Agent):
    name = "missing_authority_detector"
    ctx_keys = ["issues", "case_hits", "statute_hits"]
    system_prompt = f"""You are the Missing Authority Detector. For each legal issue from the petition analysis,
decide whether the retrieved database results provide adequate authority (a statute plus at least one binding
precedent). Flag issues that appear unsupported and describe what kind of authority should be found
(Supreme Court / High Court / statutory provision). Only reference cases that exist in the provided hits.

{GROUNDING_RULES}

Respond ONLY with JSON:
{{"unsupported_propositions": [{{"issue": "...", "severity": "high|medium|low",
"what_to_search": "suggested search query", "why_needed": "..." }}],
"adequately_supported_issues": ["..."]}}"""
