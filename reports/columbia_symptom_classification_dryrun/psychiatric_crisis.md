# psychiatric_crisis — QUARANTINED, not part of the symptom vocabulary
#
# These terms are excluded from vocabulary/symptoms.py entirely and never
# will be added there. They indicate acute suicidal/homicidal ideation or
# intent, which this service routes through the crisis path (rules/crisis.py,
# CLAUDE.md > Non-negotiable safety rules #9) — not through symptom
# extraction or RAG. A term like this reaching extract_symptoms/RAG would
# mean an acute crisis got treated as a routine triage input instead of
# stopping analysis and surfacing the crisis marker.
#
# Kept here, separately, so a mental-health-qualified reviewer can decide
# which (if any) inform rules/crisis.py's own pattern list — see the
# PLACEHOLDER DATA warning in that file. This list is a candidate source
# for that review, not itself reviewed or approved content.
#

# psychiatric_crisis (1 entries)

- [17] feeling hopeless — Feeling hopeless indicates significant emotional distress, fitting the psychiatric crisis definition.
