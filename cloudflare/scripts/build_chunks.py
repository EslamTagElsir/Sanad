"""
يولّد cloudflare/src/data/*.json من مصادر المشروع (يُشغَّل يدويًا عند تغيير data/ أو
docs/company_policies.pdf ثم يُرفع الناتج إلى git):

    python cloudflare/scripts/build_chunks.py

Worker لا يقدر يشغّل PyMuPDF، فاستخراج الـ PDF والتقطيع يتمّان هنا بنفس دوال
common.py بالضبط. أما ترميز (embedding) القطع فيتم داخل الـ Worker عبر Workers AI.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cloudflare" / "src" / "data"
sys.path.insert(0, str(ROOT))

import common  # noqa: E402


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    chunks = (common.build_kb_chunks(common.load_kb())
              + common.build_ticket_chunks(common.load_past_tickets())
              + common.build_manual_chunks(common.load_policy_manual_sections()))
    (OUT / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for name in ("golden_set.json", "golden_translations.json"):
        shutil.copy(ROOT / "data" / name, OUT / name)
    prompts = {
        "system_prompt": common.SANAD_SYSTEM_PROMPT,
        "clarify_prompt": common.CLARIFY_PROMPT.format(topics="%%TOPICS%%", question="%%QUESTION%%", max_q=common.MAX_CLARIFYING_QUESTIONS),
        "sensitive_regex": common._SENSITIVE_QUESTION_RE.pattern,
        "source_kind": common.SOURCE_KIND,
        "max_clarifying_questions": common.MAX_CLARIFYING_QUESTIONS,
    }
    (OUT / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=1), encoding="utf-8")
    by_type = {}
    for c in chunks:
        by_type[c["source_type"]] = by_type.get(c["source_type"], 0) + 1
    print(f"{len(chunks)} chunks -> {OUT / 'chunks.json'} {by_type}")


if __name__ == "__main__":
    main()
