"""Unit tests for ContextAssembler token budgeting and deterministic truncation priority."""

import pytest
from core.interview.context import ContextAssembler


def test_context_assembler_under_token_budget():
    """Verify assembled prompt strictly adheres to the 1500 token budget."""
    assembler = ContextAssembler(max_tokens=1500)
    spec = {
        "interviewer_persona": "Tech Lead",
        "tone": "rigorous",
        "knowledge_scope": ["Distributed Systems"],
    }
    stage = {
        "stage_id": "stage_2",
        "name": "Technical",
        "goals": ["Probe concurrency"],
    }
    profile = {
        "strengths": ["Python"],
        "claims": ["Built distributed lock"],
    }
    turns = [
        {"role": "interviewer", "content": "Tell me about Redis."},
        {"role": "candidate", "content": "I used redlock."},
    ]

    prompt = assembler.assemble_prompt(
        spec=spec, stage=stage, profile=profile, recent_turns=turns
    )
    assert len(prompt) > 0
    token_est = assembler.estimate_tokens(prompt)
    assert token_est <= 1500


def test_context_assembler_truncation_drops_old_turns_first():
    """Verify that when budget is exceeded, old dialogue turns are dropped first.
    
    The latest turn (current Q&A) must be preserved.
    """
    assembler = ContextAssembler(max_tokens=250)
    spec = {
        "interviewer_persona": "Staff Engineer",
        "tone": "direct",
        "knowledge_scope": ["Architecture"],
    }
    stage = {
        "stage_id": "stage_1",
        "name": "System Design",
        "goals": ["Evaluate scalability tradeoffs"],
    }
    profile = {
        "strengths": ["Cloud Architecture"],
    }
    turns = [
        {"role": "interviewer", "content": "Question 1: " + "X" * 200},
        {"role": "candidate", "content": "Answer 1: " + "Y" * 200},
        {"role": "interviewer", "content": "Question 2: " + "Z" * 200},
        {"role": "candidate", "content": "Answer 2: " + "W" * 200},
        {"role": "interviewer", "content": "Latest Question: What about partition tolerance?"},
        {"role": "candidate", "content": "Latest Answer: Network partitions require CP or AP choice."},
    ]

    prompt = assembler.assemble_prompt(
        spec=spec, stage=stage, profile=profile, recent_turns=turns
    )
    token_est = assembler.estimate_tokens(prompt)
    assert token_est <= 250
    # Oldest turn dropped
    assert "Question 1:" not in prompt
    # Latest turn preserved
    assert "Latest Question:" in prompt or "partition tolerance" in prompt
    assert "Latest Answer:" in prompt or "Network partitions" in prompt


def test_context_assembler_truncation_drops_research_second():
    """Verify that after old turns are trimmed, low-confidence research is dropped."""
    assembler = ContextAssembler(max_tokens=250)
    spec = {
        "interviewer_persona": "Engineering Director",
        "tone": "professional",
        "knowledge_scope": ["Fullstack"],
    }
    stage = {
        "stage_id": "stage_1",
        "name": "Design",
        "goals": ["Explore API design"],
    }
    profile = {
        "strengths": ["REST API"],
    }
    company_intel = {
        "company": "TechGlobal",
        "culture": "Collaborative",
        "low_confidence_notes": ["Unverified rumor: " + "R" * 400],
        "auxiliary_research": ["Extra note: " + "N" * 400],
    }
    turns = [
        {"role": "interviewer", "content": "Current question: Design an idempotent payment API."},
        {"role": "candidate", "content": "Current answer: I would use UUID idempotency keys."},
    ]

    prompt = assembler.assemble_prompt(
        spec=spec,
        stage=stage,
        profile=profile,
        company_intel=company_intel,
        recent_turns=turns,
    )
    token_est = assembler.estimate_tokens(prompt)
    assert token_est <= 250
    # Research dropped to satisfy budget
    assert "Unverified rumor" not in prompt
    assert "Extra note" not in prompt
    # Non-droppable core remains intact
    assert "Engineering Director" in prompt
    assert "Explore API design" in prompt


def test_context_assembler_truncation_drops_noncritical_profile_third():
    """Verify non-critical profile detail is dropped if budget still exceeded."""
    assembler = ContextAssembler(max_tokens=200)
    spec = {
        "interviewer_persona": "Principal Architect",
        "tone": "rigorous",
        "knowledge_scope": ["Distributed Systems"],
    }
    stage = {
        "stage_id": "stage_tech",
        "name": "Technical Depth",
        "goals": ["Verify consensus algorithms"],
    }
    profile = {
        "strengths": ["Raft consensus"],
        "claims": ["Claim A: " + "C" * 200, "Claim B: " + "D" * 200],
        "hobbies": ["Chess", "Hiking", "Brewing"],
        "metadata": {"certifications": ["Cert 1", "Cert 2", "Cert 3"]},
    }
    turns = [
        {"role": "interviewer", "content": "Explain Raft leader election."},
        {"role": "candidate", "content": "Nodes elect a leader via randomized election timeouts."},
    ]

    prompt = assembler.assemble_prompt(
        spec=spec, stage=stage, profile=profile, recent_turns=turns
    )
    token_est = assembler.estimate_tokens(prompt)
    assert token_est <= 200
    # Secondary profile details dropped
    assert "Claim A:" not in prompt
    assert "hobbies" not in prompt
    # Core strengths and goals preserved
    assert "Raft" in prompt
    assert "Principal Architect" in prompt


def test_context_assembler_never_drops_persona_tone_stage_and_current_qa():
    """Verify that persona, tone, stage goals, and current Q&A are strictly inviolable."""
    assembler = ContextAssembler(max_tokens=100)
    spec = {
        "interviewer_persona": "Strict Screener",
        "tone": "probing_and_sharp",
        "knowledge_scope": ["Algorithms"],
    }
    stage = {
        "stage_id": "s_core",
        "name": "Coding",
        "goals": ["Assess dynamic programming intuition"],
    }
    profile = {
        "strengths": ["Python"],
        "claims": ["Lots of claims " * 20],
    }
    turns = [
        {"role": "interviewer", "content": "Question: How do you solve knapsack?"},
        {"role": "candidate", "content": "Answer: 2D table or 1D rolling array with reverse loop."},
    ]

    prompt = assembler.assemble_prompt(
        spec=spec, stage=stage, profile=profile, recent_turns=turns
    )
    # Persona
    assert "Strict Screener" in prompt
    # Tone
    assert "probing_and_sharp" in prompt
    # Stage goals
    assert "Assess dynamic programming intuition" in prompt
    # Current Q&A
    assert "How do you solve knapsack?" in prompt
    assert "2D table or 1D rolling array" in prompt


def test_estimate_tokens_empty_and_text():
    """Verify estimate_tokens returns 0 for empty and positive for non-empty text."""
    assembler = ContextAssembler(max_tokens=1500)
    assert assembler.estimate_tokens("") == 0
    assert assembler.estimate_tokens(None) == 0
    assert assembler.estimate_tokens("Hello world") >= 1


def test_context_assembler_truncates_huge_candidate_turn_under_budget():
    """Verify that a massive candidate response is truncated to strictly fit under 1500 tokens."""
    assembler = ContextAssembler(max_tokens=1500)
    spec = {
        "interviewer_persona": "Senior Architect",
        "tone": "thorough",
        "knowledge_scope": ["Distributed Storage"],
    }
    stage = {
        "stage_id": "storage_design",
        "name": "Storage Engine",
        "goals": ["Evaluate LSM tree vs B-Tree tradeoffs"],
    }
    # 10,000 characters would be ~2500 tokens on its own
    huge_answer = "LSM trees provide append-only sequential writes and compaction... " * 150
    turns = [
        {"role": "interviewer", "content": "Explain write amplification in LSM trees."},
        {"role": "candidate", "content": huge_answer},
    ]

    prompt = assembler.assemble_prompt(spec=spec, stage=stage, recent_turns=turns)
    token_est = assembler.estimate_tokens(prompt)
    assert token_est <= 1500
    assert "... [truncated]" in prompt
    assert "Senior Architect" in prompt
    assert "Storage Engine" in prompt

