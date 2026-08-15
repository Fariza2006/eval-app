"""
evaluate.py
------------
Checkpoint 2: Avtomatlaşdırılmış qiymətləndirmə skripti (LLM-as-judge)

RAG sistemini (Həftə 2) test dəstindəki suallarla sınayır və hər cavabı
"LLM-as-judge" üsulu ilə skorlayır: sual, gözlənilən cavab və real cavab
başqa bir LLM sorğusuna göndərilir, model bunların semantik uyğun olub-olmadığını
qiymətləndirir.

NİYƏ DƏQİQ (EXACT MATCH) UYĞUNLUQ DEYİL:
RAG-ın cavabları sərbəst mətndir (məs. "24 iş günü" əvəzinə "İşçilər ildə 24 gün
ödənişli məzuniyyətə haqq qazanır" yaza bilər) - bunlar məzmunca eynidir, amma
mətn olaraq fərqlidir. Ona görə açıq-uclu suallar üçün LLM-as-judge istifadə edilir.

QEYD (LLM-as-judge qərəzliliyi): bu üsulun məlum məhdudiyyətləri var - judge modeli
uzun cavabları daha "keyfiyyətli" hesab edə bilər, ya da öz ifadə tərzinə bənzəyən
cavabları üstün tuta bilər. Bu, avtomatlaşdırılmış skorlara kor-koranə güvənməməyin
səbəbidir - Checkpoint 4-də bəzi nəticələr əl ilə də yoxlanılıb.
"""

import json
import time
from llm_client import chat
from rag_pipeline import answer_with_citations
from structured_output_helper import parse_json_response


JUDGE_SYSTEM_PROMPT = """Sən qiymətləndirici (judge) süni intellektsən. Sənin işin
bir sualın gözlənilən cavabı ilə real (sistem tərəfindən verilmiş) cavabı müqayisə
etmək və bunların MƏZMUNCA (semantik) uyğun olub-olmadığını qiymətləndirməkdir.

QAYDALAR:
1. Mətnin dəqiq eyni olması TƏLƏB OLUNMUR - əsas odur ki, əsas fakt/məzmun düzgün olsun.
2. Əgər gözlənilən cavab "sənədlərdə yoxdur" kimi bir şeydirsə, real cavab da
   AÇIQ ŞƏKİLDƏ "bilmirəm/yoxdur/tapılmadı" mənasında olmalıdır ki, DÜZGÜN sayılsın.
   Əgər real cavab uydurma bir fakt verirsə (halbuki gözlənilən "yoxdur" idi), bu SƏHVDİR.
3. Cavabını YALNIZ JSON formatında ver: {"correct": true/false, "reasoning": "qısa izah"}
"""


def judge_answer(question: str, expected_answer: str, actual_answer: str, max_retries: int = 2) -> dict:
    """
    LLM-as-judge: sual, gözlənilən cavab və real cavabı müqayisə edib
    {"correct": bool, "reasoning": str} qaytarır.
    """
    judge_prompt = f"""SUAL: {question}

GÖZLƏNİLƏN CAVAB: {expected_answer}

REAL CAVAB (sistemdən): {actual_answer}

Bu real cavab gözlənilən cavabla məzmunca uyğundurmu? Yuxarıdakı formatda JSON qaytar."""

    for attempt in range(max_retries):
        raw = chat(system_prompt=JUDGE_SYSTEM_PROMPT, user_prompt=judge_prompt, temperature=0.0)
        parsed = parse_json_response(raw)
        if parsed and "correct" in parsed:
            return {"correct": bool(parsed["correct"]), "reasoning": parsed.get("reasoning", "")}
        time.sleep(1)

    # Judge özü etibarlı JSON qaytara bilmədi - bunu aydın "naməlum" kimi işarələyirik
    # (avtomatik "səhv" kimi qeyd etmirik, çünki bu, real uğursuzluq deyil, judge-in problemi ola bilər)
    return {"correct": None, "reasoning": "Judge etibarlı JSON qaytara bilmədi"}


def run_evaluation(test_cases: list[dict], top_k: int = 2, verbose: bool = True) -> list[dict]:
    """
    Verilmiş test dəstini RAG sistemi üzərində işlədir və hər nəticəni judge ilə skorlayır.

    Return: hər sual üçün {"id", "question", "expected", "actual", "correct", "reasoning",
                            "latency_seconds", "category"} siyahısı
    """
    results = []

    for case in test_cases:
        start = time.time()
        rag_result = answer_with_citations(case["question"], top_k=top_k)
        elapsed = time.time() - start

        actual_answer = rag_result["answer"]

        judge_result = judge_answer(case["question"], case["expected_answer"], actual_answer)

        result = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "expected": case["expected_answer"],
            "actual": actual_answer,
            "correct": judge_result["correct"],
            "reasoning": judge_result["reasoning"],
            "latency_seconds": round(elapsed, 2),
        }
        results.append(result)

        if verbose:
            status = "✅" if judge_result["correct"] else ("❓" if judge_result["correct"] is None else "❌")
            print(f"{status} [{case['id']}] {case['question'][:50]}...")
            if not judge_result["correct"]:
                print(f"    Gözlənilən: {case['expected_answer']}")
                print(f"    Alındı: {actual_answer[:100]}")
                print(f"    Judge izahı: {judge_result['reasoning']}")

    return results


if __name__ == "__main__":
    from test_set import DEV_SET

    print("=== CHECKPOINT 2: AVTOMATLAŞDIRILMIŞ QİYMƏTLƏNDİRMƏ (DEV_SET üzərində) ===\n")
    results = run_evaluation(DEV_SET)

    correct_count = sum(1 for r in results if r["correct"] is True)
    print(f"\n\n=== YEKUN: {correct_count}/{len(results)} sual düzgün cavablandı ===")
