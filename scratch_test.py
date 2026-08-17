# scratch_test.py
from prompts.base import build_prompt
from state import merge_symptoms

p = build_prompt("_safety_preamble")
print(len(p))
print("تشخيص" in p)     # True = الملف انقرأ صح
print("الأزمة" in p)