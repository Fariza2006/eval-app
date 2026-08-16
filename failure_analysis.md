# Checkpoint 4: Kök-Səbəb Analizi — 3 Uğursuzluq Halı

Bu sənəd real testlərdə (Həftə 2 və Həftə 4) aşkarlanan **3 fərqli növ** uğursuzluğu sənədləşdirir. Hər hal üçün: nə baş verdi, NİYƏ baş verdi (kök-səbəb) və bu, hansı kateqoriyaya aiddir (zəif retrieval / zəif prompt / qiymətləndirmə metodologiyasının məhdudiyyəti).

---

## Hal 1: Zəif Retrieval (embedding modelin məhdudiyyəti)

**Sual:** *"Uzaqdan işləmək üçün nə etməliyəm?"* (Həftə 2-dəki `vector_store.py` testindən)

**Nə baş verdi:** Sistem ən oxşar 2 chunk kimi **chunk #4 (Texniki Avadanlıq)** və **chunk #0 (Ümumi Məlumat)**-ı qaytardı. Halbuki düzgün cavab **chunk #1**-də (İş Saatları və Uzaqdan İşləmə bölməsi) idi — bu chunk top-2-yə **düşmədi**.

```
Sual: Uzaqdan işləmək üçün nə etməliyəm?
  #1 (məsafə=0.9376, chunk=4): TEXNİKİ AVADANLIQ...
  #2 (məsafə=1.0365, chunk=0): ÜMUMİ MƏLUMAT...
  (chunk=1, doğru cavab olan "UZAQDAN İŞLƏMƏ" bölməsi, top-2-də YOXDUR)
```

**Kök-səbəb: ZƏİF RETRIEVAL.** İstifadə olunan embedding modeli (`sentence-transformers/all-MiniLM-L6-v2`) kiçik və sürətlidir, amma az sözlü/qısa sorğularda semantik fərqləri həmişə dəqiq ayırd edə bilmir. Əlavə olaraq, test sənədi çox kiçikdir (cəmi 5 chunk) — bu, modelin "kontekst zənginliyi" ilə köməkçi ayırd etmə imkanını azaldır.

**Mümkün həll:** daha böyük/güclü embedding modeli (`bge-large`, OpenAI `text-embedding-3`), daha böyük sənəd toplusu, ya da hybrid search (keyword + vector).

---

## Hal 2: Zəif Prompt (struktur qaydalara tam əməl edilməməsi)

**Sual:** *"Şirkətin baş direktoru (CEO) kimdir?"* (Həftə 2-dəki `rag_pipeline.py` — `answer_with_citations()` testindən)

**Nə baş verdi:** Model cavabın MƏTNİNDƏ düzgün davrandı ("TechNova MMC-nin CEO-su haqqında məlumat yoxdur"), AMMA eyni zamanda JSON-un `sources` sahəsində **2 mənbəyə istinad etdi** (boş massiv `[]` olmalı idi, çünki sistem promptunda açıq deyilib: *"Əgər kontekstdə cavab yoxdursa, sources boş olsun"*).

```json
{"answer": "...CEO-su haqqında məlumat yoxdur.", "sources": [{"chunk_id": 0}, {"chunk_id": 4}]}
```

**Kök-səbəb: ZƏİF PROMPT/MODEL MƏHDUDİYYƏTİ.** Kiçik open-source model (Llama-3.1-8B) çoxqaydalı, struktur JSON təlimatlarına **100% əməl etmir** — mətn hissəsində düzgün davranır, amma struktur hissədə (JSON sahələri) qaydanı unudur. Bu, sistem promptunun aydınlığından çox, modelin ölçüsü/instruction-following bacarığı ilə bağlıdır.

**Tətbiq olunan həll:** Həftə 2-də bunun üçün **əlavə, LLM-dən asılı olmayan qoruma** əlavə etdik — `distance_threshold` yoxlaması: əgər çəkilmiş chunk-ların məsafəsi çox yüksəkdirsə, LLM-i heç çağırmadan sistem özü "yoxdur" cavabı verir (bax Həftə 2 Checkpoint 6).

---

## Hal 3: Qiymətləndirmə Metodologiyasının Məhdudiyyəti (LLM-as-judge etibarsızlığı)

**Sual:** *"Şirkətdə neçə işçi çalışır?"* (`dev_02`, Həftə 4-dəki `evaluate.py` real testindən)

**Nə baş verdi:** RAG sistemi **düzgün** cavab verdi: *"Hazırda 120-dən çox işçi çalışır"* (gözlənilən: "120-dən çox"). Bunlar **məzmunca eynidir**. Amma judge model bunu SƏHV kimi qiymətləndirdi, izahında məntiqsiz/ziddiyyətli arqument yazaraq (iki fərqli işə salındığı test icrasında iki fərqli, hər ikisi məntiqsiz izah verdi):

```
1-ci icra: "...bu, gözlənilən cavabla məzmunca uyğun deyil" (səbəbsiz)
2-ci icra: "...'Hazırda' və '120-dən çox' məzmunca fərqlidir" (məntiqsiz)
```

**Kök-səbəb: SİSTEMİN ÖZÜ DEYİL, QİYMƏTLƏNDİRMƏ METODUNUN MƏHDUDİYYƏTİ.** Bu, RAG sisteminin uğursuzluğu DEYİL — sistem düzgün cavab verib. Problem **LLM-as-judge**-in özündədir: kiçik modellər bəzən səthi mətn fərqlərinə (məs. "Hazırda" sözünün olub-olmaması) həddindən artıq həssas ola bilər və bunu semantik fərq kimi yozur.

**Nəticə:** bu, README-də əvvəlcədən xəbərdarlıq etdiyimiz "LLM-as-judge qərəzliliyi" riskinin **canlı sübutudur** və avtomatlaşdırılmış qiymətləndirmə skorlarına kor-koranə güvənilməməsinin səbəbini göstərir — əl ilə yoxlama (bu sənəddəki kimi) hələ də vacibdir.

---

## Xülasə cədvəli

| Hal | Sual tipi | Kök-səbəb kateqoriyası | Sistem, yoxsa qiymətləndirmə problemi? |
|---|---|---|---|
| 1 | Uzaqdan iş | Zəif retrieval (kiçik embedding model) | Sistem problemi |
| 2 | CEO (hallucination) | Zəif prompt/model instruction-following | Sistem problemi (qismən, LLM-dən asılı olmayan qoruma ilə azaldılıb) |
| 3 | İşçi sayı | LLM-as-judge etibarsızlığı | Qiymətləndirmə metodologiyası problemi |
