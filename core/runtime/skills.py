"""Skill registry: versioned markdown playbooks and templates selected by code, not by the model.

Layout under the skills root:
    playbooks/<role_family>/<stage_type>.md     interviewer stage playbooks ("any" is the fallback)
    templates/<name>.md                          brief / report templates

Each file starts with front matter:
    ---
    name: technical
    version: 1.0.0
    ---


Luồng sử dụng: Registry nội dung markdown của ứng dụng, được API khởi tạo cho speaker/brief và các caller khác.
Đọc playbook theo họ vai trò và loại stage, hoặc template theo tên; trả Skill kèm version/hash
để truy vết nội dung đã dùng. Việc chọn file do code quyết định, không nhận đường dẫn từ model.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_SKILLS_ROOT = Path(__file__).resolve().parents[2] / "skills"
MAX_SKILL_CHARS = 8000


class SkillValidationError(ValueError):
    """Raised when a skill file is malformed."""


@dataclass(frozen=True)
class Skill:
    """A loaded skill file."""

    name: str
    version: str
    content: str
    sha256: str
    path: str

    @property
    def ref(self) -> str:
        """Stable reference recorded in events and decisions."""
        return f"{self.name}@{self.version}#{self.sha256[:12]}"


def _parse(path: Path) -> Tuple[Dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise SkillValidationError(f"{path.name}: missing front matter")
    try:
        _, header, body = text.split("---", 2)
    except ValueError as e:
        raise SkillValidationError(f"{path.name}: unterminated front matter") from e
    meta: Dict[str, str] = {}
    for line in header.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    if not meta.get("version"):
        raise SkillValidationError(f"{path.name}: front matter must declare a version")
    body = body.strip()
    if len(body) > MAX_SKILL_CHARS:
        raise SkillValidationError(f"{path.name}: exceeds {MAX_SKILL_CHARS} characters")
    return meta, body


class SkillRegistry:
    """Loads and serves skills; selection is deterministic and recorded by hash."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root or DEFAULT_SKILLS_ROOT)
        self._playbooks: Dict[Tuple[str, str], Skill] = {}
        self._templates: Dict[str, Skill] = {}
        # Nạp ngay khi tạo registry: front matter lỗi được báo ở khởi động thay vì giữa lượt.
        self.load()

    def _load_file(self, path: Path, default_name: str) -> Skill:
        meta, body = _parse(path)
        return Skill(
            name=meta.get("name", default_name),
            version=meta["version"],
            content=body,
            sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            path=path.relative_to(self.root).as_posix(),
        )

    def load(self) -> None:
        """(Re)load every skill under the root. Raises SkillValidationError on malformed files."""
        self._playbooks.clear()
        self._templates.clear()
        playbooks_dir = self.root / "playbooks"
        if playbooks_dir.exists():
            for path in sorted(playbooks_dir.glob("*/*.md")):
                family, stage_type = path.parent.name, path.stem
                self._playbooks[(family, stage_type)] = self._load_file(path, f"{family}/{stage_type}")
        templates_dir = self.root / "templates"
        if templates_dir.exists():
            for path in sorted(templates_dir.glob("*.md")):
                self._templates[path.stem] = self._load_file(path, path.stem)

    def get_playbook(self, role_family: Optional[str], stage_type: Optional[str]) -> Optional[Skill]:
        """Playbook for a role family and stage type, falling back to the `any` family."""
        stage = stage_type or "technical"
        # Fallback chỉ đổi họ vai trò sang any, vẫn giữ đúng loại stage đang được yêu cầu.
        return self._playbooks.get((role_family or "any", stage)) or self._playbooks.get(("any", stage))

    def get_template(self, name: str) -> Optional[Skill]:
        """Template by name (e.g. `interview_brief`, `report`)."""
        return self._templates.get(name)

    def list(self) -> List[str]:
        """References of every loaded skill."""
        return sorted([s.ref for s in self._playbooks.values()] + [s.ref for s in self._templates.values()])
