"""Template loader and safety-preamble injection for LLM prompts.

Prompt wording lives in prompts/templates/*.txt (CLAUDE.md > Conventions) —
this module is the only place that reads those files and assembles them
into the string handed to llm_client.call_llm().
"""

from __future__ import annotations

from pathlib import Path
from string import Template

__all__ = ["build_prompt"]

TEMPLATES_DIR = Path(__file__).parent / "templates"
SAFETY_PREAMBLE_NAME = "_safety_preamble"


def _load_template(name: str, **variables: str) -> str:
    """Read prompts/templates/{name}.txt and substitute $variables into it.

    Uses string.Template ($var) rather than str.format ({var}) because
    templates routinely contain literal JSON braces.

    Rejects any $-prefixed token the caller didn't supply a variable for,
    instead of silently leaving it in the output or letting substitute()
    fail with a confusing KeyError. In practice this is a tripwire against
    pasting Pydantic's model_json_schema() output into a template — its
    $ref/$defs/$schema keys collide with Template's placeholder syntax.
    That's not something to escape around: schemas are never pasted into
    templates at all (CLAUDE.md > Conventions) — structured output goes
    through the provider's native schema parameter via llm_client, and a
    template that needs to show an example uses a short hand-written
    Arabic one instead.

    Private on purpose: reading a template on its own skips the safety
    preamble. build_prompt() is the only sanctioned way to assemble a
    prompt — there is no bypass flag, and there shouldn't be one.
    """
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

    The only public entry point in this module. Every node must call this
    rather than reading a template directly, so the safety rules from
    CLAUDE.md reach the model on every single call regardless of which
    node or prompt is involved.
    """
    preamble = _load_template(SAFETY_PREAMBLE_NAME)
    body = _load_template(name, **variables)
    return f"{preamble}\n\n{body}"
