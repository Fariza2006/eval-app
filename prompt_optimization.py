"""
prompt_optimization.py
-------------------------
Checkpoint 5: Müəyyən edilmiş uğursuzluq kateqoriyasını (Hal 2 - failure_analysis.md:
"zəif prompt/model instruction-following", JSON-da sources qaydasına əməl edilməməsi)
FEW-SHOT NÜMUNƏ əlavə edərək düzəltmək və ƏVVƏL/SONRA müqayisəsi aparmaq.

VACİB (test dəsti çirklənməsinin qarşısı): bu təkmilləşdirmə YALNIZ DEV_SET-dəki
müşahidəyə (Həftə 2-nin CEO sualı) əsasən aparılıb. Yekun müqayisə isə
HELD_OUT_TEST_SET-dəki (əvvəllər HEÇ VAXT görülməmiş) bir hallucination sualı
üzərində edilir ki, nəticə dövri/etibarsız olmasın.
"""

import json
from llm_client import chat
from rag_pipeline import CITATION_SYSTEM_PROMPT, build_rag_prompt
from vector_store import search
from structured_output_helper import parse_json_response


# --- TƏKMİLLƏŞDİRİLMİŞ PROMPT: few-shot nümunə əlavə olunub ---
IMPROVED_CITATION_SYSTEM_PROMPT = CITATION_SYSTEM_PROMPT + """

NÜMUNƏ (bu formatı DƏQIQ təqlid et):

Sual: "Şirkətin ofisi hansı ölkədədir?" (fərz edək ki, bu, kontekstdə yoxdur)
Düzgün cavab formatı:
{"answer": "Bu barədə sənədlərdə məlumat tapılmadı.", "sources": []}

DİQQƏT: yuxarıdakı nümunədə "answer" sahəsi "yoxdur" desə də, "sources" siyahısı
BOŞDUR ([]) - çünki heç bir real mənbə istifadə olunmayıb. Bu qaydanı HƏR ZAMAN
tətbiq et: cavab "yoxdur" olduqda, sources HƏMİŞə boş olmalıdır."""


def get_raw_json_response(question: str, system_prompt: str, top_k: int = 2) -> dict:
    """
    answer_with_citations()-dan fərqli olaraq, distance_threshold qoruması OLMADAN
    birbaşa LLM-dən JSON cavab alır - bu, YALNIZ prompt-un özünün effektivliyini
    təcrid olunmuş şəkildə test etmək üçündür (sistemin digər qoruma qatlarını keçərək).
    """
    retrieved_chunks = search(question, top_k=top_k)
    user_prompt = build_rag_prompt(question, retrieved_chunks)
    raw_answer = chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2)
    parsed = parse_json_response(raw_answer)
    return {"raw": raw_answer, "parsed": parsed}


def check_sources_rule_followed(parsed: dict | None) -> bool:
    """Yoxlayır: cavab 'yoxdur' mənasındadırsa, sources boşdurmu?"""
    if not parsed:
        return False
    answer_text = (parsed.get("answer") or "").lower()
    says_not_found = any(kw in answer_text for kw in ["yoxdur", "tapılmadı", "məlumat yoxdur", "bilmirəm"])
    sources = parsed.get("sources", [])
    if says_not_found:
        return len(sources) == 0
    return True  # "yoxdur" demirsə, bu qayda üçün tətbiq edilmir


if __name__ == "__main__":
    # ƏVVƏL/SONRA testi: HELD_OUT_TEST_SET-dəki bir hallucination sualı üzərində
    # (bu sual təkmilləşdirmə zamanı HEÇ VAXT görülməyib - test dəsti çirklənməsi yoxdur)
    from test_set import HELD_OUT_TEST_SET

    held_out_hallucination_q = next(
        q for q in HELD_OUT_TEST_SET if q["category"] == "hallucination"
    )
    question = held_out_hallucination_q["question"]

    print(f"HELD-OUT TEST SUALI (əvvəllər görülməyib): {question}\n")

    print("=== ƏVVƏL (orijinal prompt) ===")
    before = get_raw_json_response(question, CITATION_SYSTEM_PROMPT)
    print(f"Xam JSON: {before['parsed']}")
    before_ok = check_sources_rule_followed(before["parsed"])
    print(f"'sources boş olmalı' qaydasına əməl edildi mi? {'✅ BƏLİ' if before_ok else '❌ XEYR'}")

    print("\n=== SONRA (few-shot nümunə ilə təkmilləşdirilmiş prompt) ===")
    after = get_raw_json_response(question, IMPROVED_CITATION_SYSTEM_PROMPT)
    print(f"Xam JSON: {after['parsed']}")
    after_ok = check_sources_rule_followed(after["parsed"])
    print(f"'sources boş olmalı' qaydasına əməl edildi mi? {'✅ BƏLİ' if after_ok else '❌ XEYR'}")

    print(f"\n\n=== NƏTİCƏ ===")
    print(f"Əvvəl: {'✅' if before_ok else '❌'}  ->  Sonra: {'✅' if after_ok else '❌'}")
