"""
evaluation/translate_golden.py — ترجمات LLM محفوظة لأسئلة المجموعة الذهبية
--------------------------------------------------------------------------
وقت التشغيل، الثقة تُحسب على أعلى تشابه بين رسالة العميل وترجمتها بالـ LLM
(راجع stage2_hybrid/rag.py). لكي يتدرّب نموذج الثقة على نفس التوزيع، يحتاج
ترجمات لأسئلة المجموعة الذهبية بنفس الـ prompt — نحفظها هنا مرة واحدة بدل
استدعاء LLM عند كل تشغيل للخدمة (أبطأ، غير حتمي، ويستهلك الحصة المجانية).

شغّله بعد إضافة أسئلة جديدة لـ data/golden_set.json (يترجم الناقص فقط):
    python -m evaluation.translate_golden
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from stage2_hybrid.confidence import load_golden, TRANSLATIONS_PATH
from stage2_hybrid.rag import translate_query_for_retrieval


def main():
    cache = json.loads(TRANSLATIONS_PATH.read_text(encoding="utf-8")) if TRANSLATIONS_PATH.exists() else {}
    missing = [it for it in load_golden() if not cache.get(it["id"])]
    for it in missing:
        cache[it["id"]] = translate_query_for_retrieval(it["question"])
        print(f"{it['id']}: {cache[it['id']]!r}", flush=True)
        TRANSLATIONS_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"تمت ترجمة {len(missing)} سؤال؛ الإجمالي {len(cache)}.")


if __name__ == "__main__":
    main()
