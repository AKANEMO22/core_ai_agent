"""Orchestrator notes: the two LLM-editable sections of the interview notebook.

Notes are capped per section and may not quote the candidate verbatim (more than 12 consecutive
words). Candidate speech belongs in the code-rendered evidence log, so injected instructions in
an answer cannot become part of the orchestrator's trusted memory.
"""

import re
import time
from typing import Any, Dict, Iterable, List, Set, Tuple

from core.runtime.tooling import RuntimeTool
from core.workspace.session import SessionWorkspace

NOTE_SECTIONS: Dict[str, int] = {"observations": 1200, "open_threads": 600}
NOTES_PATH = "orchestration/notes.json"
QUOTE_NGRAM = 13  # more than 12 consecutive words
_WORD = re.compile(r"\w+", re.UNICODE)


def _ngrams(words: List[str], n: int) -> Set[Tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def contains_verbatim_quote(text: str, sources: Iterable[str], n: int = QUOTE_NGRAM) -> bool:
    """True if `text` shares a run of `n` consecutive words with any source (case-insensitive)."""
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < n:
        return False
    note_grams = _ngrams(words, n)
    for source in sources:
        source_words = [w.lower() for w in _WORD.findall(source)]
        if note_grams & _ngrams(source_words, n):
            return True
    return False


class NotesStore:
    """Persists notebook note sections with a change history."""

    def __init__(self, workspace: SessionWorkspace) -> None:
        self.workspace = workspace

    def load(self) -> Dict[str, Any]:
        if self.workspace.exists(NOTES_PATH):
            data = self.workspace.read_json(NOTES_PATH)
            data = data.get("payload", data)
            return {"sections": dict(data.get("sections", {})), "history": list(data.get("history", []))}
        return {"sections": {name: "" for name in NOTE_SECTIONS}, "history": []}

    def candidate_texts(self) -> List[str]:
        texts: List[str] = []
        for path in self.workspace.list_artifacts("turns"):
            if path.name.endswith("_candidate.json"):
                rel = path.relative_to(self.workspace.root_path).as_posix()
                try:
                    texts.append(str(self.workspace.read_json(rel).get("content", "")))
                except Exception:
                    continue
        return texts

    def update(self, section: str, text: str, author: str = "orchestrator") -> Dict[str, Any]:
        """Validate and store a section. Returns {"status": ...}."""
        if section not in NOTE_SECTIONS:
            return {"status": "invalid", "error": f"section must be one of {sorted(NOTE_SECTIONS)}"}
        cap = NOTE_SECTIONS[section]
        if len(text) > cap:
            return {"status": "invalid", "error": f"section '{section}' is limited to {cap} characters"}
        if contains_verbatim_quote(text, self.candidate_texts()):
            return {
                "status": "invalid",
                "error": "notes must paraphrase; verbatim candidate quotes belong in the evidence log",
            }
        notes = self.load()
        notes["sections"][section] = text
        notes["history"].append({"section": section, "author": author, "at": time.time(), "length": len(text)})
        notes["history"] = notes["history"][-200:]
        self.workspace.write_artifact(NOTES_PATH, artifact_type="notebook_notes", payload=notes)
        return {"status": "ok", "section": section, "length": len(text)}


class UpdateNotesTool(RuntimeTool):
    """Replace one LLM-editable notebook section."""

    def __init__(self, workspace: SessionWorkspace) -> None:
        self.store = NotesStore(workspace)
        super().__init__(
            name="update_notes",
            description=(
                "Replace a notebook section: 'observations' (hypotheses about the candidate, max 1200 chars) "
                "or 'open_threads' (topics to revisit, max 600 chars). Paraphrase; never quote the candidate."
            ),
            func=self.update,
            param_schema={
                "type": "object",
                "properties": {
                    "section": {"type": "string", "enum": sorted(NOTE_SECTIONS)},
                    "text": {"type": "string", "maxLength": max(NOTE_SECTIONS.values())},
                },
                "required": ["section", "text"],
                "additionalProperties": False,
            },
        )

    async def update(self, section: str, text: str, **_: Any) -> Dict[str, Any]:
        return self.store.update(section, text)
