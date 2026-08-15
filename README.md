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
├── .env.example
├── .gitignore
└── README.md
```
