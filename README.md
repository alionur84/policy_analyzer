# KAZ — Kuzey Afrika Analiz Sistemi

Dış politika karar destek prototipi. Kuzey Afrika ülkelerine (Mısır, Libya, Tunus, Fas, Cezayir) ilişkin Türkçe haber arşivini analiz eder; politika notlarıyla karşılaştırarak yapay zeka destekli raporlar üretir.

---

## Özellikler

- **Dashboard** — Haber akışı, ülke/kaynak dağılımı, haber yoğunluğu ısı haritası, unigram/bigram/trigram terim analizi
- **Karşılaştırma** — İki dönem veya iki ülke grubunu yan yana karşılaştırma; terim farklılıkları, ısı haritaları, opsiyonel ton analizi
- **Rapor Üretimi** — 4 aşamalı LLM zinciri: tematik haber haritası → politika notu karşılaştırması → olası seyirler → nihai rapor. PDF ve Markdown formatında indirilebilir.
- **Kullanım Kılavuzu** — Uygulama içi interaktif tutorial

---

## Kurulum

```bash
git clone https://github.com/KULLANICI/kaz.git
cd kaz
pip install -r requirements.txt
```

---

## Klasör Yapısı

```
kaz/
├── app.py
├── requirements.txt
├── kaz_tutorial.html
├── data/
│   └── north_africa_data_2020-20260315.csv
├── politika_notlari/
│   ├── misir/
│   │   └── misir_bilgi_notu.pdf
│   ├── libya/
│   ├── tunus/
│   ├── fas/
│   └── cezayir/
├── fonts/
│   ├── DejaVuSans.ttf
│   ├── DejaVuSans-Bold.ttf
│   └── DejaVuSans-Oblique.ttf
└── stopwords/
    └── stopwords.txt
```

### CSV Formatı

| Sütun | Tip | Açıklama |
|---|---|---|
| `hash_id` | string | Benzersiz haber kimliği |
| `url` | string | Kaynak URL |
| `source` | string | Yayın organı adı |
| `title` | string | Haber başlığı |
| `text_full` | string | Haber tam metni |
| `word_count` | int | Kelime sayısı |
| `date` | datetime | Yayın tarihi (`YYYY-MM-DD`) |
| `country_name` | string | Ülke adı (Türkçe) |

### Politika Notları

Her ülke için `politika_notlari/[ulke_klasoru]/` altına PDF olarak yerleştirilir. Sistem başlangıçta otomatik yükler. Birden fazla not desteklenir.

| Klasör adı | Ülke |
|---|---|
| `misir` | Mısır |
| `libya` | Libya |
| `tunus` | Tunus |
| `fas` | Fas |
| `cezayir` | Cezayir |

---

## Çalıştırma

### Yerel

```bash
streamlit run app.py
```

OpenAI API key için `.streamlit/secrets.toml` oluşturun (bu dosyayı `.gitignore`'a ekleyin):

```toml
OPENAI_API_KEY = "sk-..."
APP_PASSWORD   = "sifreniz"
```

### Streamlit Community Cloud

1. Repoyu GitHub'a push edin
2. [share.streamlit.io](https://share.streamlit.io) → New app → repo ve `app.py` seçin
3. Advanced settings → Secrets bölümüne ekleyin:

```toml
OPENAI_API_KEY = "sk-..."
APP_PASSWORD   = "sifreniz"
```

---

## Konfigurasyon

`app.py` dosyasının üstündeki sabitler:

| Sabit | Varsayılan | Açıklama |
|---|---|---|
| `DATA_PATH` | `./data/north_africa_data_2020-20260315.csv` | Haber arşivi yolu |
| `POLICY_DIR` | `./politika_notlari` | Politika notları kök klasörü |
| `STOPWORDS_PATH` | `./stopwords/stopwords.txt` | Ek stopwords dosyası |
| `MAX_HABER_LLM` | `50` | LLM'e gönderilecek maksimum haber sayısı |
| `ENABLE_OLLAMA` | `False` | `True` yapılırsa yerel Ollama seçeneği aktif olur |

---

## Gereksinimler

```
streamlit
openai
pymupdf
pandas
plotly
requests
reportlab
markdown
```

Python 3.10+ önerilir.

---

## PDF Raporu için Font

Türkçe karakter desteği için DejaVu Sans fontları `fonts/` klasöründe bulunmalıdır. [DejaVu Fonts](https://dejavu-fonts.github.io) adresinden indirilebilir.

Gerekli dosyalar: `DejaVuSans.ttf`, `DejaVuSans-Bold.ttf`, `DejaVuSans-Oblique.ttf`

Font bulunamazsa sistem Helvetica'ya düşer — PDF üretilmeye devam eder ancak Türkçe karakterler görünmeyebilir.

---

## Notlar

- Rapor üretimi GPT-4o ile ortalama 1–3 dakika sürer
- Seçilen dönem 50 haberden fazlaysa sistem en temsili haberleri otomatik olarak seçer
- Karşılaştırma sekmesindeki ton analizi opsiyoneldir; ek LLM çağrısı gerektirir
- Uygulama yalnızca Türkçe yayın organlarından derlenen veriyi destekler
