
from __future__ import annotations

from pathlib import Path
from string import Template

__all__ = ["build_prompt"]

TEMPLATES_DIR = Path(__file__).parent / "templates"
SAFETY_PREAMBLE_NAME = "_safety_preamble"


def _load_template(name: str, **variables: str) -> str:
    """Load a named template and substitute any variables into it."""
    path = TEMPLATES_DIR / f"{name}.txt"
    text = path.read_text(encoding="utf-8").strip()
    template = Template(text)

    unknown = [token for token in template.get_identifiers() if token not in variables]
    if unknown:
        tokens = ", ".join(f"${token}" for token in unknown)
        raise ValueError(
            f"{name}.txt references {tokens} with no matching variable supplied. "
            "Templates must not contain pasted JSON schema output "
            "($ref/$defs/...) or any other unresolved $-token. "
            "See CLAUDE.md > Conventions."
        )

    return template.substitute(**variables)


def build_prompt(name: str, **variables: str) -> str:
    """Load a named template with the safety preamble injected ahead of it.
    """
    preamble = _load_template(SAFETY_PREAMBLE_NAME)
    body = _load_template(name, **variables)
    return f"{preamble}\n\n{body}"
