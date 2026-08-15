# AI Çıxışının Qiymətləndirilməsi + Kiçik Model Adaptasiyası

Bu layihə **Həftə 2**-dəki RAG sistemini (`company_handbook.txt` üzərində) qiymətləndirir: test dəsti qurur, avtomatlaşdırılmış qiymətləndirmə skripti yazır, metrikləri izləyir, uğursuzluqları kök-səbəb analizi ilə sənədləşdirir və prompt-u təkmilləşdirir.

**İstifadə olunan texnologiyalar:** Həftə 2-nin RAG pipeline-ı (`rag_pipeline.py`, `vector_store.py`, `embeddings.py`, `ingest.py`, `llm_client.py`) — bax həmin fayllar.

## Quraşdırma

Həftə 2 ilə eynidir: `pip install chromadb requests python-dotenv`, `.env` faylında `HF_API_TOKEN`.

---

# Checkpoint 1: Test Dəsti (18 sual/gözlənilən-cavab cütü)

`test_set.py` **18 sual** ehtiva edir (tələb olunan 15-20 aralığında), 3 kateqoriyaya bölünüb:

| Kateqoriya | Say | Təsvir |
|---|---|---|
| `normal` | 12 | Sənəddə birbaşa, açıq cavabı olan sadə suallar |
| `edge_case` | 3 | Dolayı ifadə (paraphrase), chunk-sərhəd riski, qeyri-müəyyən sual |
| `hallucination` | 3 | Sənəddə **ümumiyyətlə olmayan** sual — gözlənilən cavab "yoxdur" |

## VACİB: Test dəsti çirklənməsinin qarşısını almaq üçün bölgü

Tapşırıqda qeyd olunan **"test dəsti çirklənməsi" trick-i**nin qarşısını almaq üçün, test dəsti **iki AYRI, üst-üstə düşməyən** hissəyə bölünüb:

- **`DEV_SET`** (12 sual) — Checkpoint 4-5-də prompt-u təkmilləşdirmək üçün istifadə olunacaq
- **`HELD_OUT_TEST_SET`** (6 sual) — **təkmilləşdirmə zamanı HEÇ VAXT istifadə olunmur**, yalnız Checkpoint 5-dəki **yekun, qərəzsiz "əvvəl/sonra" müqayisəsi** üçün saxlanılır

Bu bölgü olmadan, "təkmilləşmə" nəticələri etibarsız olardı (eyni sualları həm tənzimləmə, həm yoxlama üçün istifadə etmək dövri validasiyadır).

## İşlətmək

```bash
python test_set.py
```

## Fayl strukturu

```
eval-app/
├── documents/company_handbook.txt   # Həftə 2-dən
├── ingest.py, embeddings.py, vector_store.py, llm_client.py, rag_pipeline.py, structured_output_helper.py  # Həftə 2-dən
├── test_set.py                       # Checkpoint 1: test dəsti (18 sual, dev/held-out bölgüsü)
├── evaluate.py                        # Checkpoint 2: avtomatlaşdırılmış qiymətləndirmə (LLM-as-judge)
├── .env.example
├── .gitignore
└── README.md
```

---

# Checkpoint 2: Avtomatlaşdırılmış Qiymətləndirmə Skripti

`evaluate.py` RAG sistemini test dəstindəki hər sualla sınayır və cavabları **LLM-as-judge** üsulu ilə skorlayır.

## Niyə dəqiq (exact match) uyğunluq deyil, LLM-as-judge?

RAG-ın cavabları sərbəst mətndir. Məsələn, gözlənilən cavab `"24 iş günü"` olsa da, sistem `"İşçilər ildə 24 gün ödənişli məzuniyyətə haqq qazanır"` yaza bilər — bunlar **məzmunca eynidir**, mətn olaraq fərqlidir. Ona görə başqa bir LLM sorğusu bu ikisinin semantik uyğunluğunu qiymətləndirir.

## ⚠️ LLM-as-judge qərəzliliyi (vacib məhdudiyyət)

Tapşırıqda xüsusi qeyd olunan riskə uyğun olaraq, bunu **açıq şəkildə qeyd edirik**: LLM-as-judge üsulunun məlum qərəzləri var:
- **Uzunluq qərəzi** — judge modeli daha uzun/ətraflı cavabları "daha keyfiyyətli" hesab edə bilər, hətta məzmun eyni olsa belə.
- **Öz-ifadə qərəzi** — judge model öz "danışıq tərzinə" bənzəyən cavabları üstün tuta bilər.
- Ona görə avtomatlaşdırılmış skorlara **kor-koranə güvənilmir** — Checkpoint 4-də bəzi "səhv" işarələnən nəticələr əl ilə də yoxlanılıb ki, bunun həqiqətən sistem xətası, yoxsa judge-in özünün səhvi olduğu ayırd edilsin.

## Judge-in xüsusi qaydası (hallüsinasiya sualları üçün)

Judge-ə açıq təlimat verilib: əgər gözlənilən cavab "yoxdur"-dursa, real cavab da açıq şəkildə "bilmirəm/yoxdur" mənasında olmalıdır ki, DÜZGÜN sayılsın. Əgər sistem bu tip sualda uydurma fakt versə, bu, AVTOMATİK SƏHV kimi qeydə alınır (judge-in "uzun cavab = yaxşı" qərəzinin bu kritik halda təsir etməməsi üçün xüsusi qayda).

## İşlətmək

```bash
python evaluate.py
```

## Nümunə çıxış (format)

```
✅ [dev_01] Şirkət neçənci ildə təsis edilib?...
✅ [dev_06] İllik ödənişli məzuniyyət neçə gündür?...
❌ [dev_11] Şirkətin baş direktoru (CEO) kimdir?...
    Gözlənilən: Sənədlərdə yoxdur / məlumat yoxdur
    Alındı: TechNova MMC-nin baş direktoru Anar Məmmədovdur.
    Judge izahı: Sistem uydurma fakt verib, sənəddə bu məlumat yoxdur.

=== YEKUN: 10/12 sual düzgün cavablandı ===
```

---

# Checkpoint 3: İzlənilən Metriklər

`metrics.py` `evaluate.py`-in nəticələrini alıb, aqreqasiya edilmiş metrikləri hesablayır:

- **Accuracy/pass-rate** — düzgün cavabların faizi (judge "naməlum" saydığı hallar bu hesabdan xaric edilir ki, rəqəm süni şəkildə aşağı düşməsin)
- **Orta cavab müddəti (latency)** — hər sualın RAG pipeline-dan cavab almaq üçün çəkdiyi orta vaxt
- **Orta token xərci** — `llm_client.py`-ə əlavə etdiyimiz token izləmə mexanizmi (`_usage_log`) vasitəsilə hesablanır; hər API çağırışının `prompt_tokens`, `completion_tokens`, `estimated_cost` məlumatı toplanır
- **Kateqoriya üzrə accuracy** — `normal`/`edge_case`/`hallucination` sualları ayrı-ayrı qiymətləndirilir ki, sistemin HANSI tip sualda daha zəif olduğu aydın olsun

## İşlətmək

```bash
python metrics.py
```

## Nümunə çıxış (format)

```
==================================================
METRİKLƏR HESABATI
==================================================
Ümumi sual sayı:      12
Düzgün:                9
Səhv:                  2
Naməlum (judge xətası): 1
Accuracy (pass-rate):  81.8%
Orta cavab müddəti:    2.3 saniyə
Orta giriş tokeni:     650.0
Orta çıxış tokeni:     45.0
Orta cəmi token:       695.0
Orta xərc (sorğu üçün): $0.000015

Kateqoriya üzrə accuracy:
  normal: 7/8 (87.5%)
  edge_case: 1/2 (50.0%)
  hallucination: 1/2 (50.0%)
```

Bu format aydın göstərir ki, sistem **normal** suallarda güclü, amma **edge_case** və **hallucination** suallarında nisbətən zəifdir — bu, Checkpoint 4-dəki kök-səbəb analizinin başlanğıc nöqtəsidir.
