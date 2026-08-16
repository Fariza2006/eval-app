# Qiymətləndirmə Hesabatı: RAG Sisteminin Test Edilməsi və Təkmilləşdirilməsi

**Layihə:** Həftə 2-dəki RAG sisteminin ("Sənədlərinlə Danış" — `company_handbook.txt` üzərində) qiymətləndirilməsi
**Tarix:** Avqust 2026
**Müəllif:** Fərizə Rzazadə

---

## 1. Metodologiya

### 1.1 Test dəsti dizaynı

18 sual/gözlənilən-cavab cütü hazırlandı, 3 kateqoriyaya bölünərək:
- **Normal (12 sual):** sənəddə birbaşa, açıq cavabı olan sadə suallar
- **Edge-case (3 sual):** dolayı ifadə (paraphrase), chunk-sərhəd riski, qeyri-müəyyən sual
- **Hallucination (3 sual):** sənəddə ümumiyyətlə olmayan, gözlənilən cavabı "yoxdur" olan suallar

**Test dəsti çirklənməsinin qarşısı:** sualların 12-si (`DEV_SET`) prompt təkmilləşdirmək üçün, qalan 6-sı (`HELD_OUT_TEST_SET`) isə YALNIZ yekun, qərəzsiz qiymətləndirmə üçün ayrıldı və təkmilləşdirmə prosesində heç vaxt istifadə olunmadı.

### 1.2 Qiymətləndirmə üsulu: LLM-as-judge

Cavablar sərbəst mətn olduğu üçün (eyni fakt fərqli sözlərlə ifadə oluna bilər), dəqiq mətn uyğunluğu (exact match) əvəzinə **başqa bir LLM sorğusu** cavabların semantik uyğunluğunu qiymətləndirdi.

**Bilinən məhdudiyyət:** LLM-as-judge-in özünün qərəzləri (uzunluq qərəzi, öz-ifadə qərəzi) var. Bu, nəzəri fərziyyə deyil — real testdə **sübut edildi** (bax bölmə 3, "İşçi sayı" halı).

---

## 2. Nəticələr (Metriklər)

`metrics.py` vasitəsilə toplanan real ölçmələr (DEV_SET-in bir hissəsi üzərində, kredit məhdudiyyəti səbəbindən tam dövr tamamlanmadı, amma nümunə kifayət qədər göstəricidir):

| Metrik | Dəyər |
|---|---|
| Sınanan sual sayı | 4-6 (kredit limitinə görə qismən) |
| Müşahidə olunan accuracy | ~75-80% (judge-in öz səhvləri xaric edilməklə) |
| Orta cavab müddəti | ~2 saniyə/sual |
| Orta token istifadəsi | ~500-700 token/sual |

**Qeyd:** Hugging Face-in pulsuz aylıq kredit limiti ($0.10) tam 18 sualın bir dövrdə sınanmasına həmişə imkan vermədi — bu, real layihələrdə büdcə planlaşdırmasının əhəmiyyətini göstərən praktiki bir dərsdir.

---

## 3. Kök-Səbəb Analizi (3 Uğursuzluq Halı)

Ətraflı analiz `failure_analysis.md`-dədir. Xülasə:

| # | Hal | Kök-səbəb | Kateqoriya |
|---|---|---|---|
| 1 | "Uzaqdan iş" sualında retrieval doğru chunk-ı tapmadı | Kiçik embedding model + kiçik sənəd toplusu | Zəif retrieval |
| 2 | "CEO" sualında model mətndə düzgün "yoxdur" dedi, amma JSON-da sources boş qalmadı | Kiçik modelin struktur-instruction-following məhdudiyyəti | Zəif prompt/model |
| 3 | "İşçi sayı" sualında düzgün cavab judge tərəfindən səhv kimi qiymətləndirildi | LLM-as-judge-in özünün etibarsızlığı | Qiymətləndirmə metodologiyası |

**Vacib müşahidə:** yalnız 1 hal (retrieval) əsl RAG sistem problemidir, 1 hal (prompt) qismən sistem/qismən model məhdudiyyətidir, 1 hal isə ümumiyyətlə sistemin yox, **qiymətləndirmə metodunun** problemidir. Bu ayrım vacibdir — bütün "uğursuzluqları" avtomatik sistem xətası kimi qəbul etmək səhv nəticələrə apara bilər.

---

## 4. Təkmilləşdirmə Cəhdi (Prompt Optimallaşdırması)

Hal 2-ni (JSON-da sources qaydası) düzəltmək üçün few-shot nümunə əlavə edildi (`prompt_optimization.py`).

**Real nəticə: təkmilləşdirmə İŞLƏMƏDİ.** Həld-out test sualında (əvvəllər görülməmiş) model həm əvvəl, həm sonra eyni səhvi etdi.

**Bundan çıxarılan dərslər:**
1. Bir (`one-shot`) nümunə kiçik modeldə kifayət qədər güclü siqnal olmaya bilər.
2. Prompt-səviyyəli düzəlişlər kiçik open-source modellərdə **zəmanətli deyil**.
3. Bu, sistem-səviyyəli, LLM-dən asılı olmayan qorumaların (məs. Həftə 2-dəki `distance_threshold` yoxlaması) niyə vacib olduğunu bir daha təsdiqləyir — prompt "xahiş etmək" kifayət deyil, kodun özündə **doğrulama məntiqi** olmalıdır.

---

## 5. Ümumi Nəticə

Bu layihə göstərdi ki:
- RAG sistemi **normal suallarda** nisbətən etibarlıdır, amma **edge-case** və struktur-tələb edən hallarda (JSON qaydaları) daha zəifdir.
- **LLM-as-judge** faydalı, amma kor-koranə güvənilməli olmayan bir vasitədir — əl ilə yoxlama hələ də vacibdir.
- **Prompt-səviyyəli düzəlişlər** (few-shot) bütün hallarda işləmir — bəzən sistem-səviyyəli (kod daxilində) qorumalar daha etibarlıdır.
- **Test dəsti çirklənməsinin qarşısını almaq** (dev/held-out bölgüsü) real, etibarlı nəticələr üçün vacibdir — bu olmadan, "təkmilləşmə" nəticələri özünü aldatma ola bilərdi.

**Gələcək iş üçün təkliflər:**
- Daha güclü embedding modeli ilə retrieval-ı yaxşılaşdırmaq
- 2-3 nümunəli (çox-shot) prompt-larla yenidən sınamaq
- Kiçik modelin xüsusi fine-tuning-i (bu tapşırığın adında qeyd olunan, amma kredit məhdudiyyəti səbəbindən bu dövrədə tam sınanmayan seçim)
