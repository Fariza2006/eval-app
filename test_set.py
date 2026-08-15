"""
test_set.py
------------
Checkpoint 1: Normal + kənar hallardan ibarət test dəsti (18 sual/gözlənilən-cavab cütü)

Bu test dəsti Həftə 2-dəki RAG sisteminin (company_handbook.txt sənədi üzərində)
qiymətləndirilməsi üçündür.

VACİB - "Test dəsti çirklənməsi" trick-inin qarşısını almaq üçün:
Test dəsti İKİ AYRI, ÜST-ÜSTƏ DÜŞMƏYƏN hissəyə bölünüb:
- DEV_SET (12 sual): prompt-u təkmilləşdirmək üçün istifadə olunur (Checkpoint 4-5)
- HELD_OUT_TEST_SET (6 sual): TƏKMİLLƏŞDİRMƏ ZAMANI HEÇ VAXT İSTİFADƏ OLUNMUR,
  yalnız YEKUN, qərəzsiz qiymətləndirmə üçün saxlanılır (Checkpoint 5-dəki
  əvvəl/sonra müqayisəsi buradan gəlməlidir, əks halda nəticə dövri/etibarsız olar)

Hər sualın kateqoriyası var:
- "normal": sənəddə birbaşa, açıq cavabı olan sadə suallar
- "edge_case": dolayı ifadə, çətin retrieval (chunk-sərhəd), qeyri-müəyyən sual
- "hallucination": sənəddə ÜMUMİYYƏTLƏ olmayan sual - gözlənilən cavab "yoxdur"
"""

DEV_SET = [
    {
        "id": "dev_01",
        "question": "Şirkət neçənci ildə təsis edilib?",
        "expected_answer": "2015",
        "category": "normal",
    },
    {
        "id": "dev_02",
        "question": "Şirkətdə neçə işçi çalışır?",
        "expected_answer": "120-dən çox",
        "category": "normal",
    },
    {
        "id": "dev_03",
        "question": "Şirkət hansı sahədə fəaliyyət göstərir?",
        "expected_answer": "maliyyə texnologiyaları (fintech)",
        "category": "normal",
    },
    {
        "id": "dev_04",
        "question": "Standart iş saatları nə vaxtdan nə vaxtadək davam edir?",
        "expected_answer": "09:00-dan 18:00-dək",
        "category": "normal",
    },
    {
        "id": "dev_05",
        "question": "Ayda neçə gün uzaqdan işləmək olar?",
        "expected_answer": "8 günə qədər",
        "category": "normal",
    },
    {
        "id": "dev_06",
        "question": "İllik ödənişli məzuniyyət neçə gündür?",
        "expected_answer": "24 iş günü",
        "category": "normal",
    },
    {
        "id": "dev_07",
        "question": "Xəstəlik məzuniyyəti neçə gündür?",
        "expected_answer": "14 gün",
        "category": "normal",
    },
    {
        "id": "dev_08",
        "question": "Əmək haqqı ayın neçəsində ödənilir?",
        "expected_answer": "hər ayın 5-də",
        "category": "normal",
    },
    {
        "id": "dev_09",
        "question": "İllik məzuniyyət neçə gündür və bu necə hesablanır (hansı dövr üzrə)?",
        "expected_answer": "24 iş günü, təqvim ili üzrə (yanvarın 1-dən dekabrın 31-dək)",
        "category": "edge_case",  # chunk-sərhəd riski - Həftə 2-dəki eyni fakt
    },
    {
        "id": "dev_10",
        "question": "Kompaniya nə vaxt yaradılıb?",  # "təsis edilib" sözünün paraphrase-i
        "expected_answer": "2015",
        "category": "edge_case",
    },
    {
        "id": "dev_11",
        "question": "Şirkətin baş direktoru (CEO) kimdir?",
        "expected_answer": "Sənədlərdə yoxdur / məlumat yoxdur",
        "category": "hallucination",
    },
    {
        "id": "dev_12",
        "question": "Şirkətdə uşaq baxımı (daycare) xidməti varmı?",
        "expected_answer": "Sənədlərdə yoxdur / məlumat yoxdur",
        "category": "hallucination",
    },
]

HELD_OUT_TEST_SET = [
    {
        "id": "held_01",
        "question": "Cümə günləri iş saatı neçədə bitir?",
        "expected_answer": "15:00",
        "category": "normal",
    },
    {
        "id": "held_02",
        "question": "Məzuniyyət günləri növbəti ilə maksimum neçə gün keçirilə bilər?",
        "expected_answer": "5 gün",
        "category": "normal",
    },
    {
        "id": "held_03",
        "question": "İllik performans qiymətləndirməsinə əsasən maksimum bonus faizi neçədir?",
        "expected_answer": "15%",
        "category": "normal",
    },
    {
        "id": "held_04",
        "question": "Yeni işə qəbul olunan işçiyə hansı texniki avadanlıq verilir?",
        "expected_answer": "noutbuk, monitor və əlavə periferik avadanlıq",
        "category": "normal",
    },
    {
        "id": "held_05",
        "question": "İş şəraiti necədir?",  # qəsdən qeyri-müəyyən/geniş sual
        "expected_answer": (
            "Qeyri-müəyyən sual - sənəddə iş saatları, uzaqdan iş, məzuniyyət kimi "
            "bir neçə fərqli bölmə var, konkret cavab tələb edilməlidir"
        ),
        "category": "edge_case",
    },
    {
        "id": "held_06",
        "question": "İşçilərə pulsuz nahar verilirmi?",
        "expected_answer": "Sənədlərdə yoxdur / məlumat yoxdur",
        "category": "hallucination",
    },
]


def all_questions() -> list[dict]:
    """Bütün 18 sualı (dev + held-out) birlikdə qaytarır - ümumi statistika üçün."""
    return DEV_SET + HELD_OUT_TEST_SET


if __name__ == "__main__":
    print(f"DEV_SET: {len(DEV_SET)} sual")
    print(f"HELD_OUT_TEST_SET: {len(HELD_OUT_TEST_SET)} sual")
    print(f"CƏMİ: {len(all_questions())} sual\n")

    for group_name, group in [("DEV_SET", DEV_SET), ("HELD_OUT_TEST_SET", HELD_OUT_TEST_SET)]:
        print(f"\n=== {group_name} ===")
        categories = {}
        for item in group:
            categories[item["category"]] = categories.get(item["category"], 0) + 1
        print(f"Kateqoriyalar: {categories}")
        for item in group:
            print(f"  [{item['id']}] ({item['category']}) {item['question']}")
