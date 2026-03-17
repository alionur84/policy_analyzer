"""
KAZ - Kuzey Afrika Analiz Sistemi — v0.1
Dış Politika Karar Destek Sistemi
"""

import re
import json
import os
from collections import Counter
from datetime import date
from pathlib import Path

import fitz
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════
# KONFİGÜRASYON
# ═══════════════════════════════════════════════════════════════════

DATA_PATH      = "./data/north_africa_data_2020-20260315.csv"
POLICY_DIR     = "./politika_notlari"
STOPWORDS_PATH = "./stopwords/stopwords.txt"

ULKELER = ["Mısır", "Libya", "Tunus", "Fas", "Cezayir"]

# LLM'e gönderilecek maksimum haber sayısı (context window & maliyet kontrolü)
MAX_HABER_LLM = 50

# Lokal LLM (Ollama) seçeneği — False yapılırsa sidebar'da görünmez
ENABLE_OLLAMA = False

ULKE_KLASOR = {
    "misir":   "Mısır",
    "libya":   "Libya",
    "tunus":   "Tunus",
    "fas":     "Fas",
    "cezayir": "Cezayir",
}

AMBER  = "#F59E0B"
TEAL   = "#14B8A6"
ROSE   = "#F43F5E"
VIOLET = "#8B5CF6"
SKY    = "#38BDF8"
LIME   = "#84CC16"
ORANGE = "#FB923C"
SLATE  = "#94A3B8"
COLORS = [AMBER, TEAL, ROSE, VIOLET, SKY, LIME, ORANGE, SLATE]

PLOTLY_LAYOUT = dict(
    plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
    font=dict(family="Inter", size=11, color="#CBD5E1"),
)
AXIS_STYLE = dict(gridcolor="#1E293B", zerolinecolor="#334155", color="#94A3B8")

DEFAULT_STOPWORDS = {
    "ve","ile","bir","bu","da","de","ki","için","olan","olarak","den","dan",
    "in","mi","mı","mu","mü","ya","ne","o","çok","daha","en","var","yok",
    "gibi","kadar","sonra","önce","her","hiç","bazı","tüm","ise","ama",
    "fakat","ancak","veya","hem","bunu","bunun","şunu","onun","bunları",
    "haberi","haber","yıl","ay","gün","tarih","oldu","edildi","yapıldı",
    "açıkladı","belirtti","söyledi","dedi","göre","üzere","dolayı",
    "türkiye","türk","son","yeni","bin","açıklamada","yapılan","olduğunu",
    "arasında","başkanı","bakanı","cumhurbaşkanı","hafter","etti","eden",
    "şu","biz","siz","onlar","ben","sen","bana","sana","ona","bize",
    "size","onlara","benden","senden","ondan","bizden","sizden","onlardan",
}

# ═══════════════════════════════════════════════════════════════════
# PROMPT'LAR  (PROMPT_2 KESİNLİKLE DEĞİŞTİRİLMEDİ)
# ═══════════════════════════════════════════════════════════════════

SISTEM_PROMPT = """Sen Türkiye Dışişleri Bakanlığı için çalışan kıdemli bir bölgesel analiz uzmanısın. \
Görevin; haber verileri ve politika notlarını sentezleyerek karar alıcılara yönelik analitik raporlar üretmektir.

Temel ilkeler:
- Olgusal iddialar mutlaka sağlanan haber datasına dayanmalı. Dayanağı olmayan iddia üretme.
- Kesinlik hiyerarşisine riayet et: "teyit edilmiştir" / "raporlanmıştır" / "değerlendirilmektedir" / "öngörülmektedir" ayrımını koru.
- Yüzde olasılık verme. Olasılık sıralaması için "en muhtemel", "muhtemel", "daha az muhtemel" ifadelerini kullan.
- Türkçe yaz. Teknik terimler dışında yabancı dil kullanma.
- Raporun her bölümü bağımsız okunabilir olmalı.
- Üslup: Resmi ama okunabilir. Akademik jargondan kaçın."""

PROMPT_1 = """Aşağıda {ulke} ile ilgili {tarih_araligi} dönemine ait {haber_sayisi} haber metni verilmiştir.

## Haberler
{haberler}

Görevin:
1. Bu haberleri tematik olarak gruplandır. Her tema için bir başlık ve o temayı temsil eden 2-3 haberi referans göster. Her kısmın sonuna "kaynak" yazarak URL'leri hyperlink ile ekle).
2. Dönemin en kritik 3 gelişmesini belirle, bu gelişmelerin ne olduğunu açık ve net biçimde kısaca anlat ve kısaca gerekçele. Bu gelişmeleri haberlerden doğrudan çıkar ve gerekirse alıntıla.
3. Dönemde öne çıkan aktörleri listele: kim ne yaptı, ne dedi?
4. Haberlerin kaynak dağılımına dikkat et — hangi konular hangi yayın organları tarafından öne çıkarılmış?
5. Doğrudan verili ülke ile ilgili olmayan haberleri, spor, kültür ve yaşam haberlerini göz ardı et.

Bu aşamada yorum yapma, yalnızca tespit et ve haritalandır. Çıktıyı yapılandırılmış başlıklarla sun."""

PROMPT_2 = """Aşağıda iki girdi verilmiştir:
(A) {ulke} için tematik haber haritası:
{haber_haritasi}

(B) {ulke} Politika Notu:
{politika_notu}

Görevin:
1. Politika notundaki bilgiler dahilinde Türkiye perspektifinden bakarak dönemin haberlerini değerlendir. Hangi gelişmeler Türkiye'nin savunduğu pozisyonlarla örtüşüyor, hangisi sürtüşme yaratıyor? Bu gelişmelere haberlerden kaynak ver. 
2. Haberlerde doğrudan olmayan hiç bir konu ile ilişkili yorum yapma, sadece politika notunda bulunan bilgileri, eğer bu konular ile ilişkili haber yoksa tekrarlama.
3. Varsa çelişkili haberler veya birbiriyle örtüşmeyen anlatıları belirle. 

KURAL — Bu kurala harfiyen uy:
Her iddia doğrudan yukarıdaki haber metnine dayanmalıdır. Haberde geçmeyen hiçbir konu hakkında yorum, çıkarım veya tahmin üretme. "Haberlerde bu konuda doğrudan bir gelişme yer almasa da..." gibi ifadeler YASAKTIR.

Çıktıyı aşağıdaki başlıklarla sun. Bir başlık altında habere dayanan içerik yoksa o başlığı tamamen atla, boş bırakma:

### Örtüşen Dinamikler
Yalnızca haberlerde doğrudan yer alan ve politika notuyla örtüşen gelişmeler. Her madde için varsa url ile kaynak haber belirt.

### Sürtüşme Alanları
Yalnızca haberlerde doğrudan yer alan ve Türkiye'nin pozisyonlarıyla çelişen gelişmeler. Her madde için varsa url ile kaynak haber belirt.

### Ayrışan Okumalar
Farklı haber kaynakları aynı olayı farklı perspektiflerden ele almışsa kısa değerlendirme. Haber yoksa bu başlığı atla."""

PROMPT_4 = """Aşağıdaki analiz girdileri verilmiştir:
{onceki_analizler}

Görevin iki bölümlü bir çıktı üretmek:

BÖLÜM A — Olası Seyirler
Önümüzdeki 6-12 ay için {ulke}-Türkiye ilişkilerinde veya {ulke}'nin iç/bölgesel dinamiklerinde 3 farklı seyir tanımla.
Seyirleri olasılık sıralamasına göre sun (en muhtemel → daha az muhtemel). Her seyir için:
- Seyri tetikleyecek koşul veya gelişme nedir?
- Türkiye'ye etkisi ne olur?
- Önerilen yaklaşım nedir?

BÖLÜM B — Politika Önerileri
Yukarıdaki analizden türeyen somut, numaralı, kısa öneriler sun. Her öneri tek cümle olabilir. Genel geçer tavsiyelerden kaçın — spesifik ve uygulanabilir ol."""

PROMPT_5 = """Aşağıdaki analiz bölümleri verilmiştir:

BÖLÜM 1 — Tematik Haber Haritası:
{haber_haritasi}

BÖLÜM 2 — Analitik Değerlendirme:
{analitik}

BÖLÜM 3 — Olası Seyirler ve Öneriler:
{seyirler}

Bu bölümleri aşağıdaki rapor formatında birleştir. Yönetici özetini EN SON yaz — tüm analizi gördükten sonra.

---
# {ulke} Analiz Raporu | {tarih_araligi}

## Yönetici Özeti
(maksimum 200 kelime — karar alıcı için, teknik detay yok)

## Dönemsel Gelişmeler
(Tematik haber haritasından — tematik gruplandırma, kronoloji zorunlu değil)

## Analitik Değerlendirme
(Analitik değerlendirmeden — örtüşen dinamikler, sürtüşme alanları, ayrışan okumalar)

## Olası Seyirler
(Seyirler bölümünden — en muhtemelden daha az muhtemele)

## Politika Önerileri
(Öneriler bölümünden — numaralı, somut)

---
*Rapor {haber_sayisi} haber kaydına dayanmaktadır. Dönem: {tarih_araligi}. Kaynak dağılımı: {kaynaklar}*
---

Üslup: Resmi ama okunabilir. Akademik jargondan kaçın."""


# ═══════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════

@st.cache_data
def load_stopwords() -> set:
    try:
        with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
            words = {line.strip().lower() for line in f if line.strip()}
        return DEFAULT_STOPWORDS | words
    except FileNotFoundError:
        return DEFAULT_STOPWORDS


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return _prepare_df(df)


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    df["date_only"] = df["date"].dt.date
    df["ym"]        = df["date"].dt.to_period("M").astype(str)
    df["yw"]        = df["date"].dt.strftime("%G-W%V")
    df["year"]      = df["date"].dt.year
    df["combined_text"] = (
        df["title"].fillna("") + " " +
        df["title"].fillna("") + " " +
        df["text_full"].fillna("")
    )
    return df


@st.cache_data
def load_policy_notes_cached(policy_dir: str) -> dict:
    notlar: dict[str, dict[str, str]] = {u: {} for u in ULKELER}
    base = Path(policy_dir)
    if not base.exists():
        return notlar
    for klasor, ulke in ULKE_KLASOR.items():
        ulke_path = base / klasor
        if not ulke_path.exists():
            continue
        for pdf_path in sorted(ulke_path.glob("*.pdf")):
            try:
                doc   = fitz.open(str(pdf_path))
                metin = "".join(s.get_text() for s in doc)
                doc.close()
                notlar[ulke][pdf_path.name] = metin.strip()
            except Exception:
                pass
    return notlar


def tr_lower(s: str) -> str:
    """Türkçe büyük-küçük harf dönüşümü — İ→i, I→ı."""
    return s.replace("İ", "i").replace("I", "ı").lower()


def top_terms(df_sub: pd.DataFrame, n: int = 20, stopwords: set = None,
              max_gram: int = 2) -> list:
    """
    Unigram + bigram (+ trigram, max_gram=3) frekans sayımı.
    Stopword içeren ngramlar filtrelenir.
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    words = Counter()
    for text in df_sub["combined_text"].fillna(""):
        tokens = [
            w for w in re.findall(r'\b[a-züğışöçA-ZÜĞİŞÖÇ]{2,}\b', tr_lower(text))
            if not w.isdigit()
        ]
        # Unigram
        for w in tokens:
            if w not in stopwords and len(w) >= 3:
                words[w] += 1
        # Bigram
        if max_gram >= 2:
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i+1]
                if w1 not in stopwords and w2 not in stopwords:
                    words[f"{w1} {w2}"] += 1
        # Trigram
        if max_gram >= 3:
            for i in range(len(tokens) - 2):
                w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
                if w1 not in stopwords and w2 not in stopwords and w3 not in stopwords:
                    words[f"{w1} {w2} {w3}"] += 1
    return words.most_common(n)


def top_terms_by_gram(df_sub: pd.DataFrame, stopwords: set = None,
                      n: int = 20) -> dict:
    """
    Unigram, bigram ve trigram listelerini ayrı ayrı döndürür.
    Dönüş: {"uni": [(kelime, frekans)...], "bi": [...], "tri": [...]}
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    uni, bi, tri = Counter(), Counter(), Counter()
    for text in df_sub["combined_text"].fillna(""):
        tokens = [
            w for w in re.findall(r'\b[a-züğışöçA-ZÜĞİŞÖÇ]{2,}\b', tr_lower(text))
            if not w.isdigit()
        ]
        for w in tokens:
            if w not in stopwords and len(w) >= 3:
                uni[w] += 1
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i+1]
            if w1 not in stopwords and w2 not in stopwords:
                bi[f"{w1} {w2}"] += 1
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            if (w1 not in stopwords and w2 not in stopwords
                    and w3 not in stopwords):
                tri[f"{w1} {w2} {w3}"] += 1
    return {
        "uni": uni.most_common(n),
        "bi":  bi.most_common(n),
        "tri": tri.most_common(n),
    }


def temsili_haberler(df_sub: pd.DataFrame, stopwords: set, n: int = 5) -> list:
    """
    Dönemin top terimleriyle örtüşen, en temsili haberleri döndürür.
    Başlıktaki dönem terimlerinin frekans-ağırlıklı toplamı skor olarak kullanılır.
    """
    if df_sub.empty:
        return []
    donem_terimleri = dict(top_terms(df_sub, 30, stopwords))
    if not donem_terimleri:
        return []
    skorlar = []
    for _, row in df_sub.iterrows():
        baslik_lower = tr_lower(str(row.get("title", "")))
        skor = sum(
            donem_terimleri.get(w, 0)
            for w in re.findall(r'\b[a-züğışöçA-ZÜĞİŞÖÇ]{3,}\b', baslik_lower)
            if w not in stopwords
        )
        skorlar.append((skor, row))
    skorlar.sort(key=lambda x: x[0], reverse=True)
    sonuc = []
    for skor, row in skorlar[:n]:
        sonuc.append({
            "baslik": str(row.get("title", ""))[:120],
            "url":    str(row.get("url", "")),
            "kaynak": str(row.get("source", "")),
            "tarih":  str(row.get("date_only", "")),
            "skor":   skor,
        })
    return sonuc



def term_diff(terms_a, terms_b, n=15):
    dict_a, dict_b = dict(terms_a), dict(terms_b)
    total_a = sum(dict_a.values()) or 1
    total_b = sum(dict_b.values()) or 1
    diffs = []
    for w in set(dict_a) | set(dict_b):
        fa, fb = dict_a.get(w, 0) / total_a, dict_b.get(w, 0) / total_b
        diffs.append((w, fa, fb, fa - fb))
    diffs.sort(key=lambda x: x[3], reverse=True)
    return [d for d in diffs if d[3] > 0][:n], [d for d in diffs if d[3] < 0][-n:][::-1]


def ngram_grafigi(terms: list, baslik: str, renk: str,
                  key: str, yukseklik: int = 400) -> None:
    """Verilen (kelime, frekans) listesinden yatay bar chart çizer."""
    if not terms:
        st.caption(f"{baslik}: yeterli veri yok.")
        return
    tw, tv = zip(*terms)
    fig = go.Figure(go.Bar(
        x=list(tv), y=list(tw), orientation="h",
        marker_color=renk, marker_line_width=0,
        text=list(tv), textposition="outside",
        textfont=dict(color="#CBD5E1"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        margin=dict(t=10, b=10, l=max(80, max(len(w) for w in tw) * 7), r=50),
        height=yukseklik,
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE, autorange="reversed")
    st.plotly_chart(fig, use_container_width=True, key=key)


def pick_granularity(df_sub: pd.DataFrame, gran: str) -> pd.DataFrame:
    if gran == "Günlük":
        s = df_sub.groupby("date_only").size().reset_index(name="count")
        s["label"] = s["date_only"].astype(str)
    elif gran == "Haftalık":
        s = df_sub.groupby("yw").size().reset_index(name="count")
        s["label"] = s["yw"]
    elif gran == "Aylık":
        s = df_sub.groupby("ym").size().reset_index(name="count")
        s["label"] = s["ym"]
    else:
        s = df_sub.groupby("year").size().reset_index(name="count")
        s["label"] = s["year"].astype(str)
    return s


def periyot_secenekleri(df_sub: pd.DataFrame, gran: str) -> list:
    """Granüleriteye göre seçilebilir periyot listesi döndürür."""
    if gran == "Haftalık":
        return sorted(df_sub["yw"].unique().tolist())
    elif gran == "Aylık":
        return sorted(df_sub["ym"].unique().tolist())
    elif gran == "Yıllık":
        return sorted(df_sub["year"].astype(str).unique().tolist())
    else:  # Günlük
        return sorted(df_sub["date_only"].astype(str).unique().tolist())


def df_periyot_filtrele(df_sub: pd.DataFrame, gran: str, secim: list) -> pd.DataFrame:
    """Seçilen periyotlara göre dataframe filtreler."""
    if not secim:
        return df_sub
    if gran == "Haftalık":
        return df_sub[df_sub["yw"].isin(secim)]
    elif gran == "Aylık":
        return df_sub[df_sub["ym"].isin(secim)]
    elif gran == "Yıllık":
        return df_sub[df_sub["year"].astype(str).isin(secim)]
    else:
        return df_sub[df_sub["date_only"].astype(str).isin(secim)]


def periyot_ozet(df_sub: pd.DataFrame, periyot_col: str, periyot_val: str,
                 stopwords: set, n_terms: int = 5, n_titles: int = 5) -> dict:
    sub = df_sub[df_sub[periyot_col] == periyot_val]
    if sub.empty:
        return {"count": 0, "top_terms": [], "top_titles": []}
    terms  = top_terms(sub, n_terms, stopwords)
    titles = temsili_haberler(sub, stopwords, n_titles)
    return {"count": len(sub), "top_terms": terms, "top_titles": titles}


def isi_haritasi_olustur(df_sub: pd.DataFrame, gran: str, stopwords: set) -> tuple:
    periyot_col = "yw" if gran == "Haftalık" else "ym"
    pivot = df_sub.groupby(["country_name", periyot_col]).size().unstack(fill_value=0)
    if pivot.empty:
        return None, {}, periyot_col

    ulkeler    = pivot.index.tolist()
    periyotlar = pivot.columns.tolist()
    z_vals     = pivot.values.tolist()
    lookup     = {}
    hover_matrix = []

    for ulke in ulkeler:
        row_hover = []
        for per in periyotlar:
            ozet = periyot_ozet(
                df_sub[df_sub["country_name"] == ulke],
                periyot_col, per, stopwords
            )
            lookup[(ulke, per)] = ozet
            terms_str  = ", ".join(w for w, _ in ozet["top_terms"]) or "—"
            titles_str = "<br>".join(
                f"• {t['baslik']} ({t['kaynak']}, {t['tarih']})"
                for t in ozet["top_titles"]
            ) or "—"
            row_hover.append(
                f"<b>{ulke} · {per}</b><br>"
                f"Haber: <b>{ozet['count']}</b><br><br>"
                f"<b>Öne çıkan terimler:</b><br>{terms_str}<br><br>"
                f"<b>Örnek başlıklar:</b><br>{titles_str}"
            )
        hover_matrix.append(row_hover)

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=periyotlar,
        y=ulkeler,
        colorscale=[[0, "#0F172A"], [0.4, "#0E4D6E"], [0.7, TEAL], [1.0, AMBER]],
        text=z_vals,
        texttemplate="%{text}",
        textfont=dict(size=9, color="#F8FAFC"),
        hovertext=hover_matrix,
        hovertemplate="%{hovertext}<extra></extra>",
        showscale=True,
        colorbar=dict(tickfont=dict(color="#CBD5E1"), thickness=12),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(260, 65 * len(ulkeler)),
        margin=dict(t=10, b=10, l=100, r=20),
    )
    fig.update_xaxes(**AXIS_STYLE, tickangle=-45)
    fig.update_yaxes(**AXIS_STYLE)
    return fig, lookup, periyot_col


def render_isi_haritasi(df_sub: pd.DataFrame, gran: str, stopwords: set, key_prefix: str):
    """Isı haritası + detay expander (tam genişlik, kendi satırında)."""
    fig, lookup, periyot_col = isi_haritasi_olustur(df_sub, gran, stopwords)
    if fig is None:
        st.caption("Veri yetersiz.")
        return

    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_fig")

    pivot_ulkeler    = sorted(set(u for u, _ in lookup.keys()))
    pivot_periyotlar = sorted(set(p for _, p in lookup.keys()))

    with st.expander("🔍 Dönem Detayı — ülke ve dönem seçin", expanded=True):
        dc1, dc2 = st.columns(2)
        with dc1:
            sel_ulke = st.selectbox("Ülke", pivot_ulkeler, key=f"{key_prefix}_ulke")
        with dc2:
            sel_per = st.selectbox(
                "Dönem", pivot_periyotlar,
                index=len(pivot_periyotlar) - 1,
                key=f"{key_prefix}_per",
            )

        ozet = lookup.get((sel_ulke, sel_per), {})
        if not ozet or ozet["count"] == 0:
            st.info("Bu dönemde bu ülkeye ait haber yok.")
            return

        st.markdown(f"**{sel_ulke} · {sel_per} — {ozet['count']} haber**")
        if ozet["top_terms"]:
            term_str = "  ·  ".join(f"`{w}` ({n})" for w, n in ozet["top_terms"])
            st.markdown(f"**Öne çıkan terimler:** {term_str}")

        if ozet["top_titles"]:
            st.markdown("**Örnek haberler:**")
            for t in ozet["top_titles"]:
                url = t["url"]
                baslik = t["baslik"]
                kaynak = t["kaynak"]
                tarih  = t["tarih"]
                # Link ve metin ayrı kolonlarda — iç içe geçme yok
                if url and url.startswith("http"):
                    st.markdown(
                        f"- [{baslik}]({url})  "
                        f"<span style='color:#64748B;font-size:0.8rem'>"
                        f"{kaynak} · {tarih}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"- {baslik}  "
                        f"<span style='color:#64748B;font-size:0.8rem'>"
                        f"{kaynak} · {tarih}</span>",
                        unsafe_allow_html=True,
                    )


def haberleri_formatla(df: pd.DataFrame, max_chars: int = 1200) -> str:
    satirlar = []
    for _, row in df.iterrows():
        metin = str(row.get("text_full", ""))
        if len(metin) > max_chars:
            metin = metin[:max_chars] + "..."
        satirlar.append(
            f"[{row['date_only']} | {row['source']}]\n"
            f"Başlık: {row['title']}\n"
            f"URL: {row.get('url', '')}\n"
            f"Metin: {metin}"
        )
    return "\n---\n".join(satirlar)


def markdown_to_pdf(md_text: str) -> bytes:
    """
    Markdown → PDF (reportlab, DejaVu Sans, Türkçe destekli).
    Renk teması: beyaz/açık zemin — baskıya uygun.
    """
    import io
    import re as _re
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
        ListFlowable, ListItem, Image as RLImage, KeepTogether,
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
                    
    # ── Font kaydı ────────────────────────────────────────────────────
    FREG = "DejaVuSans"
    FBLD = "DejaVuSans-Bold"
    FOBL = "DejaVuSans-Oblique"
    font_base = Path("./fonts")
    try:
        reg = pdfmetrics.getRegisteredFontNames()
        if FREG not in reg:
            pdfmetrics.registerFont(TTFont(FREG, str(font_base / "DejaVuSans.ttf")))
        if FBLD not in reg:
            pdfmetrics.registerFont(TTFont(FBLD, str(font_base / "DejaVuSans-Bold.ttf")))
        if FOBL not in reg:
            pdfmetrics.registerFont(TTFont(FOBL, str(font_base / "DejaVuSans-Oblique.ttf")))
        from reportlab.lib.fonts import addMapping
        addMapping(FREG, 0, 0, FREG)
        addMapping(FREG, 1, 0, FBLD)
        addMapping(FREG, 0, 1, FOBL)
        addMapping(FREG, 1, 1, FBLD)
    except Exception:
        FREG = FBLD = FOBL = "Helvetica"

    # ── Renkler (açık zemin) ─────────────────────────────────────────
    C_AMBER   = colors.HexColor("#D97706")   # baskıda daha net amber
    C_NAVY    = colors.HexColor("#1E3A5F")
    C_DARK    = colors.HexColor("#0F172A")
    C_SLATE   = colors.HexColor("#334155")
    C_GRAY    = colors.HexColor("#64748B")
    C_LGRAY   = colors.HexColor("#F1F5F9")
    C_BORDER  = colors.HexColor("#CBD5E1")
    C_TEAL    = colors.HexColor("#0D9488")
    C_CHART   = [colors.HexColor(h) for h in [
        "#1E3A5F","#D97706","#0D9488","#7C3AED","#DC2626","#059669","#9333EA","#B45309"
    ]]

    # ── Sayfa yapısı ─────────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.5*cm,  bottomMargin=2.5*cm,
    )
    W = A4[0] - 4.4*cm   # kullanılabilir genişlik

    base = getSampleStyleSheet()

    def S(name, font=None, **kw):
        return ParagraphStyle(name, parent=base["Normal"],
                               fontName=font or FREG, **kw)

    ST = {
        "h1":     S("H1",  font=FBLD, fontSize=16, textColor=C_DARK,
                    leading=20, spaceBefore=12, spaceAfter=4),
        "h2":     S("H2",  font=FBLD, fontSize=13, textColor=C_NAVY,
                    leading=17, spaceBefore=12, spaceAfter=3),
        "h3":     S("H3",  font=FBLD, fontSize=11, textColor=C_SLATE,
                    leading=15, spaceBefore=8,  spaceAfter=2),
        "body":   S("Body",            fontSize=10, textColor=C_SLATE,
                    leading=15, spaceAfter=4),
        "bullet": S("Bul",             fontSize=10, textColor=C_SLATE,
                    leading=14, leftIndent=14, spaceAfter=2),
        "meta":   S("Meta",font=FOBL,  fontSize=8,  textColor=C_GRAY,
                    leading=12, spaceAfter=2),
    }

    def esc(t):
        return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    def inline(text):
        text = esc(text)
        text = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = _re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
        text = _re.sub(r"`(.+?)`",       r'<font name="Courier" size="9">\1</font>', text)
        text = _re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        return text

    # ── Story oluştur ────────────────────────────────────────────────
    story = []
    bullet_buffer = []

    def flush_bullets():
        if bullet_buffer:
            items = [
                ListItem(Paragraph(inline(b), ST["bullet"]),
                         leftIndent=20, bulletColor=C_AMBER)
                for b in bullet_buffer
            ]
            story.append(ListFlowable(items, bulletType="bullet",
                                       start="•", leftIndent=10))
            story.append(Spacer(1, 4))
            bullet_buffer.clear()

    # ── Rapor ana başlığı (her zaman en üstte) ────────────────────
    # md_text'ten ilk # başlığını çek
    _baslik_match = _re.search(r"^#\s+(.+)$", md_text, _re.MULTILINE)
    _rapor_basligi = _baslik_match.group(1).strip() if _baslik_match else "Analiz Raporu"

    story.append(Paragraph(
        _rapor_basligi,
        S("RaporBaslik", font=FBLD, fontSize=18, textColor=C_DARK,
          spaceBefore=0, spaceAfter=4, leading=22)
    ))
    story.append(HRFlowable(width="100%", thickness=3,
                             color=C_AMBER, spaceBefore=2, spaceAfter=16))

    # Markdown'ı parse et
    for line in md_text.split("\n"):
        stripped = line.strip()

        if _re.match(r"^-{3,}$|^\*{3,}$|^_{3,}$", stripped):
            flush_bullets()
            story.append(HRFlowable(width="100%", thickness=0.5,
                                     color=C_BORDER, spaceAfter=8, spaceBefore=8))
            continue

        m = _re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            flush_bullets()
            level = len(m.group(1))
            text  = inline(m.group(2).strip())
            key   = f"h{level}" if level <= 3 else "h3"
            if level == 1:
                story.append(Spacer(1, 6))
                story.append(HRFlowable(width="100%", thickness=2,
                                         color=C_AMBER, spaceAfter=4))
            story.append(Paragraph(text, ST[key]))
            continue

        m = _re.match(r"^[\-\*]\s+(.*)", line)
        if m:
            bullet_buffer.append(m.group(1).strip())
            continue

        m = _re.match(r"^\d+\.\s+(.*)", line)
        if m:
            bullet_buffer.append(m.group(1).strip())
            continue

        if not stripped:
            flush_bullets()
            story.append(Spacer(1, 4))
            continue

        if stripped.startswith("*") and stripped.endswith("*") and len(stripped) > 2:
            flush_bullets()
            story.append(Paragraph(inline(stripped[1:-1]), ST["meta"]))
            continue

        flush_bullets()
        story.append(Paragraph(inline(stripped), ST["body"]))

    flush_bullets()
    doc.build(story)
    return buf.getvalue()



def kaynak_dagılımı(df: pd.DataFrame) -> str:
    sayimlar = df["source"].value_counts().head(8)
    return ", ".join([f"{k} ({v})" for k, v in sayimlar.items()])


# ── LLM ──────────────────────────────────────────────────────────────

def get_client(model_tipi: str, api_key: str = "", ollama_url: str = "") -> tuple:
    if model_tipi == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        return OpenAI(api_key=key), st.session_state.get("openai_model", "gpt-4o")
    return (
        OpenAI(base_url=f"{ollama_url}/v1", api_key="ollama"),
        st.session_state.get("ollama_model", "llama3"),
    )


def llm_call(client, model, sistem, kullanici, stream=True, temperature=0.1):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sistem},
            {"role": "user",   "content": kullanici},
        ],
        temperature=temperature,
        stream=stream,
    )


def stream_write(placeholder, stream_obj) -> str:
    tam = ""
    for chunk in stream_obj:
        delta = chunk.choices[0].delta.content
        if delta:
            tam += delta
            placeholder.markdown(tam + "▌")
    placeholder.markdown(tam)
    return tam


def llm_ton_analizi(client, model, df_sub, label="") -> dict:
    titles = "\n".join(f"- {t}" for t in df_sub["title"].dropna().head(80))
    system = """Sen bir medya analisti. Verilen başlıkları analiz et.
SADECE şu JSON formatında yanıt ver, başka hiçbir şey yazma:
{"destekleyici":0,"nötr":0,"eleştirel":0,"kriz_odakli":0,"diplomatik":0,"özet":"2-3 cümle"}
Yüzdeler toplamı 100 olmalı. özet Türkçe."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Periyot: {label}\n\n{titles}"},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {"destekleyici": 0, "nötr": 0, "eleştirel": 0,
                "kriz_odakli": 0, "diplomatik": 0, "özet": f"Hata: {e}"}


def llm_karsilastirma_yorumu(client, model, stats_a, stats_b, label_a, label_b, diff_terms) -> str:
    system = "Sen bir medya analistisin. İki dönem/grup arasındaki farkları Türkçe analiz et. 3-4 paragraf."
    user   = (
        f"A ({label_a}): {stats_a['count']} haber\n"
        f"B ({label_b}): {stats_b['count']} haber\n"
        f"A'da öne çıkan terimler: {[t[0] for t in diff_terms[0][:10]]}\n"
        f"B'de öne çıkan terimler: {[t[0] for t in diff_terms[1][:10]]}\n"
        f"A ton: {stats_a.get('ton', {})}\n"
        f"B ton: {stats_b.get('ton', {})}\n"
        "Temel farkları, odak kaymasını ve öne çıkan temaları analiz et."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Yorum üretilemedi: {e}"


# ═══════════════════════════════════════════════════════════════════
# SAYFA AYARLARI & CSS
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="KAZ - Kuzey Afrika Analiz Sistemi",
    page_icon="🪿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Şifre koruması ──────────────────────────────────────────────
import hmac

def check_password():
    def password_entered():
        if hmac.compare_digest(
            st.session_state["password"],
            st.secrets["APP_PASSWORD"]
        ):
            st.session_state["authenticated"] = True
            del st.session_state["password"]
        else:
            st.session_state["authenticated"] = False

    if st.session_state.get("authenticated"):
        return True

    st.markdown("## 🔐 KAZ — Giriş")
    st.text_input("Şifre", type="password",
                  on_change=password_entered, key="password")
    if "authenticated" in st.session_state:
        st.error("Hatalı şifre.")
    return False

if not check_password():
    st.stop()
# ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}

.metric-card{
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.1);
    border-left:3px solid #F59E0B;
    border-radius:6px;padding:1rem 1.2rem;margin:0.3rem 0;
}
.metric-label{font-size:0.72rem;color:#94A3B8;text-transform:uppercase;
              letter-spacing:0.1em;margin-bottom:4px;}
.metric-value{font-size:1.7rem;font-weight:600;color:#F8FAFC;line-height:1.1;}
.metric-sub{font-size:0.8rem;color:#64748B;margin-top:4px;}

.filter-bar{
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem;
}
.filter-active{
    background:rgba(20,184,166,0.08);
    border:1px solid rgba(20,184,166,0.35);
    border-radius:6px;padding:0.5rem 0.9rem;
    font-size:0.82rem;color:#14B8A6;margin-bottom:0.8rem;
    display:inline-block;
}
.compare-header-a{background:linear-gradient(90deg,#1E3A5F,#1a2744);
    color:#38BDF8;padding:0.6rem 1rem;border-radius:4px;font-weight:600;
    margin-bottom:0.8rem;border-left:3px solid #38BDF8;}
.compare-header-b{background:linear-gradient(90deg,#3D2000,#2d1a00);
    color:#F59E0B;padding:0.6rem 1rem;border-radius:4px;font-weight:600;
    margin-bottom:0.8rem;border-left:3px solid #F59E0B;}
.section-title{font-size:1.05rem;font-weight:600;color:#F8FAFC;
    border-bottom:1px solid rgba(255,255,255,0.1);
    padding-bottom:0.4rem;margin:1.5rem 0 0.8rem 0;}
.policy-box{background:rgba(20,184,166,0.08);border:1px solid rgba(20,184,166,0.3);
    border-radius:6px;padding:0.8rem 1rem;margin:0.4rem 0;font-size:0.85rem;color:#CBD5E1;}

/* Landing page */
.landing-card{
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:10px;padding:1.4rem 1.6rem;margin-bottom:1rem;
}
.landing-title{font-size:1rem;font-weight:600;color:#F8FAFC;margin-bottom:0.5rem;}
.landing-desc{font-size:0.88rem;color:#94A3B8;line-height:1.6;}
.landing-tag{display:inline-block;background:rgba(245,158,11,0.15);color:#F59E0B;
    border-radius:4px;padding:2px 8px;font-size:0.75rem;margin-right:4px;margin-top:6px;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# VERİ YÜKLEME (sadece otomatik — manual upload yok)
# ═══════════════════════════════════════════════════════════════════

STOPWORDS = load_stopwords()

if "df" not in st.session_state:
    if Path(DATA_PATH).exists():
        with st.spinner("Veri yükleniyor..."):
            st.session_state["df"] = load_csv(DATA_PATH)
    else:
        st.session_state["df"] = None

if "policy_notes" not in st.session_state:
    st.session_state["policy_notes"] = load_policy_notes_cached(POLICY_DIR)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR — sadece model ayarları
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🪿 KAZ - Kuzey Afrika Analiz")
    st.caption("Haber Analiz Aracı")
    st.markdown("---")

    st.markdown("**⚙️ Model Ayarları**")

    if ENABLE_OLLAMA:
        secenekler = ["OpenAI (Cloud)", "Ollama (Lokal)"]
    else:
        secenekler = ["OpenAI (Cloud)"]

    model_tipi = st.radio("LLM Kaynağı", secenekler, index=0)

    if model_tipi == "OpenAI (Cloud)":
        env_key = os.environ.get("OPENAI_API_KEY", "")
        if env_key:
            st.success("API key env'den alındı ✓")
            api_key_input = env_key
        else:
            api_key_input = st.text_input("OpenAI API Key", type="password")
        st.session_state["openai_model"] = st.selectbox(
            "Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-mini"], index=0)
        st.session_state["llm_tipi"] = "openai"
        st.session_state["api_key"]  = api_key_input
    else:  # Ollama — sadece ENABLE_OLLAMA=True ise buraya gelinir
        ollama_url_input = st.text_input("Ollama URL", value="http://localhost:11434")
        try:
            resp = requests.get(f"{ollama_url_input}/api/tags", timeout=3)
            modeller = [m["name"] for m in resp.json().get("models", [])]
            st.session_state["ollama_model"] = (
                st.selectbox("Model", modeller) if modeller
                else st.text_input("Model adı", value="llama3")
            )
        except Exception:
            st.session_state["ollama_model"] = st.text_input(
                "Model adı (Ollama bağlanamadı)", value="llama3")
        st.session_state["llm_tipi"]   = "ollama"
        st.session_state["ollama_url"] = ollama_url_input

    # Veri durumu özeti
    st.markdown("---")
    st.markdown("**📊 Veri Durumu**")
    df_ss = st.session_state.get("df")
    if df_ss is not None:
        st.success(f"✓ {len(df_ss):,} haber")
        st.caption(f"{df_ss['date_only'].min()} – {df_ss['date_only'].max()}")
    else:
        st.error(f"CSV bulunamadı:\n`{DATA_PATH}`")

    pn = st.session_state.get("policy_notes", {})
    pn_count = sum(len(v) for v in pn.values())
    if pn_count:
        st.success(f"✓ {pn_count} politika notu")
        for ulke, notlar in pn.items():
            if notlar:
                st.caption(f"{ulke}: {len(notlar)} not")
    else:
        st.warning(f"Politika notu bulunamadı\n`{POLICY_DIR}/[ulke]/`")


# ═══════════════════════════════════════════════════════════════════
# VERİ YOKSA DUR
# ═══════════════════════════════════════════════════════════════════

df = st.session_state.get("df")

if df is None:
    st.title("🗺️ Kuzey Afrika Analiz Sistemi")
    st.error(f"Haber verisi bulunamadı. Beklenen konum: `{DATA_PATH}`")
    st.stop()

date_min    = df["date_only"].min()
date_max    = df["date_only"].max()
all_sources = sorted(df["source"].dropna().unique().tolist())

# ═══════════════════════════════════════════════════════════════════
# ANA BAŞLIK & SEKMELER
# ═══════════════════════════════════════════════════════════════════

st.markdown(
    f'<h1 style="font-size:1.8rem;font-weight:700;color:#F8FAFC;margin-bottom:0">'
    f'🗺️ Kuzey Afrika Analiz Sistemi</h1>'
    f'<p style="color:#64748B;margin-bottom:0.5rem">'
    f'{len(df):,} haber · {date_min} – {date_max}</p>',
    unsafe_allow_html=True,
)

tab_landing, tab_dash, tab_compare, tab_rapor = st.tabs(
    ["ℹ️ Kullanım Kılavuzu", "📊 Dashboard", "⚖️ Karşılaştırma", "📋 Rapor Üretimi"]
)


# ═══════════════════════════════════════════════════════════════════
# TAB 0 — KULLANIM KILAVUZU (landing)
# ═══════════════════════════════════════════════════════════════════

with tab_landing:
    with open("kaz_tutorial.html", "r") as f:
        st.components.v1.html(f.read(), height=900, scrolling=True)
#     st.markdown("""
# <div style="max-width:860px">
# <h2 style="color:#F8FAFC;font-size:1.4rem;margin-bottom:0.3rem">Sisteme Hoş Geldiniz</h2>
# <p style="color:#94A3B8;margin-bottom:1.8rem;font-size:0.92rem">
# Bu sistem, Kuzey Afrika'daki beş ülkeye (Mısır, Libya, Tunus, Fas, Cezayir) ilişkin Türkçe haber akışını
# analiz etmek ve politika notlarıyla karşılaştırarak Türkiye perspektifinden analitik raporlar üretmek
# amacıyla geliştirilmiştir.
# </p>
# </div>
# """, unsafe_allow_html=True)

#     c1, c2 = st.columns(2)

#     with c1:
#         st.markdown("""
# <div class="landing-card">
# <div class="landing-title">📊 Dashboard</div>
# <div class="landing-desc">
# Haber verisini keşfedin. Zaman serisi grafikleri, ülke ve kaynak dağılımları,
# en sık geçen terimler ve haber yoğunluğu ısı haritası burada bulunur.<br><br>
# Sayfanın üstündeki filtre çubuğundan <b>zaman birimi</b>, <b>dönem</b>, <b>ülke</b>
# ve <b>kaynak</b> seçip <b>Filtre Uygula</b> butonuna basın. Aktif filtre her zaman
# görünür biçimde gösterilir.<br><br>
# Isı haritasında herhangi bir hücrenin üzerine gelerek o ülkenin o dönemdeki
# gündem maddelerini görebilir; <i>Dönem Detayı</i> bölümünden örnek haberlere
# ulaşabilirsiniz.
# </div>
# <span class="landing-tag">Filtre</span>
# <span class="landing-tag">Grafikler</span>
# <span class="landing-tag">Terim Analizi</span>
# <span class="landing-tag">Isı Haritası</span>
# </div>
# """, unsafe_allow_html=True)

#         st.markdown("""
# <div class="landing-card">
# <div class="landing-title">📋 Rapor Üretimi</div>
# <div class="landing-desc">
# Seçtiğiniz ülke, dönem ve politika notlarını birleştirerek yapılandırılmış
# bir analitik rapor üretin.<br><br>
# <b>Nasıl çalışır?</b><br>
# 1. Ülke ve tarih aralığı seçin.<br>
# 2. Kullanılacak politika notlarını işaretleyin (her ülke için ayrı).<br>
# 3. <i>Rapor Üret</i> butonuna basın.<br><br>
# Sistem dört aşamalı bir LLM zinciri çalıştırır:<br>
# ① Tematik haber haritası → ② Politika notu karşılaştırması →
# ③ Olası seyirler & öneriler → ④ Nihai rapor birleştirme.<br><br>
# Tamamlanan rapor <b>.md formatında indirilebilir</b>.
# </div>
# <span class="landing-tag">LLM Analizi</span>
# <span class="landing-tag">Politika Notu</span>
# <span class="landing-tag">Rapor İndirme</span>
# </div>
# """, unsafe_allow_html=True)

#     with c2:
#         st.markdown("""
# <div class="landing-card">
# <div class="landing-title">⚖️ Karşılaştırma</div>
# <div class="landing-desc">
# İki farklı dönem veya ülke grubunu yan yana karşılaştırın.<br><br>
# Her grup için bağımsız tarih aralığı, ülke ve kaynak filtresi ayarlayabilirsiniz.
# Karşılaştırma şunları üretir:<br>
# • Haber akışı zaman serisi (A vs B)<br>
# • Kaynak dağılımı karşılaştırması<br>
# • Terim farklılıkları (hangi kelimeler A'da öne çıkmış, hangisi B'de)<br>
# • Her grup için ısı haritası<br>
# • <i>Opsiyonel:</i> LLM ton analizi ve karşılaştırmalı yorum
# </div>
# <span class="landing-tag">Dönem Karşılaştırma</span>
# <span class="landing-tag">Terim Diff</span>
# <span class="landing-tag">Ton Analizi</span>
# </div>
# """, unsafe_allow_html=True)

#         st.markdown("""
# <div class="landing-card">
# <div class="landing-title">⚙️ Teknik Notlar</div>
# <div class="landing-desc">
# <b>Veri:</b> Türkçe yayın organlarından derlenen haber arşivi.
# Mevcut dönem: <b>{date_min} – {date_max}</b><br><br>
# <b>LLM:</b> OpenAI gpt-4o modeli kullanılmaktadır.
# <br><br>
# <b>Politika notları:</b> Her ülke için birden fazla
# not desteklenir. Notlar demo için NotebookLM ile jenerik olarak oluşturulmuştur<br><br>
# </div>
# <span class="landing-tag">OpenAI / Ollama</span>
# <span class="landing-tag">PDF Politika Notları</span>
# </div>
# """.format(date_min=date_min, date_max=date_max), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════

with tab_dash:

    # ── Filtre çubuğu — 2 satır ─────────────────────────────────────
    with st.container():
        st.markdown('<div class="filter-bar">', unsafe_allow_html=True)

        # Satır 1: Zaman birimi + dönem seçimi
        fr1_c1, fr1_c2 = st.columns([1, 3])
        with fr1_c1:
            granularity = st.selectbox(
                "Zaman birimi",
                ["Aylık", "Haftalık", "Yıllık", "Günlük"],
                key="gran",
            )
        with fr1_c2:
            tum_periyotlar = periyot_secenekleri(df, granularity)
            default_per = tum_periyotlar[-12:] if len(tum_periyotlar) > 12 else tum_periyotlar
            secili_periyotlar = st.multiselect(
                "Dönem seçimi",
                tum_periyotlar,
                default=default_per,
                key="dash_periyot",
            )

        # Satır 2: Ülke + kaynak + butonlar
        fr2_c1, fr2_c2, fr2_c3, fr2_c4 = st.columns([2, 2, 1, 1])
        with fr2_c1:
            dash_ulkeler = st.multiselect("Ülke", ULKELER, key="dash_ulke")
        with fr2_c2:
            dash_sources = st.multiselect("Kaynak", all_sources, key="dash_src")
        with fr2_c3:
            st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
            filtre_uygula = st.button("▶ Filtre Uygula", type="primary", key="dash_apply", use_container_width=True)
        with fr2_c4:
            st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
            filtre_sifirla = st.button("↺ Sıfırla", key="dash_reset", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Filtre state yönetimi
    if filtre_sifirla:
        for k in ["dash_filtre_aktif", "dash_periyot_secim", "dash_ulke_secim",
                  "dash_src_secim", "dash_gran_secim"]:
            st.session_state.pop(k, None)
        st.rerun()

    if filtre_uygula or "dash_filtre_aktif" in st.session_state:
        if filtre_uygula:
            st.session_state["dash_filtre_aktif"]   = True
            st.session_state["dash_periyot_secim"]  = secili_periyotlar
            st.session_state["dash_ulke_secim"]     = dash_ulkeler
            st.session_state["dash_src_secim"]      = dash_sources
            st.session_state["dash_gran_secim"]     = granularity

        _gran    = st.session_state.get("dash_gran_secim", granularity)
        _periyot = st.session_state.get("dash_periyot_secim", secili_periyotlar)
        _ulkeler = st.session_state.get("dash_ulke_secim", [])
        _sources = st.session_state.get("dash_src_secim", [])

        dfd = df_periyot_filtrele(df, _gran, _periyot)
        if _ulkeler:
            dfd = dfd[dfd["country_name"].isin(_ulkeler)]
        if _sources:
            dfd = dfd[dfd["source"].isin(_sources)]

        # Aktif filtre etiketi
        etiketler = []
        if _periyot:
            per_str = f"{_periyot[0]}…{_periyot[-1]}" if len(_periyot) > 2 else ", ".join(_periyot)
            etiketler.append(f"📅 {per_str}")
        if _ulkeler:
            etiketler.append("🌍 " + ", ".join(_ulkeler))
        if _sources:
            etiketler.append(f"📰 {len(_sources)} kaynak")
        if etiketler:
            st.markdown(
                f'<div class="filter-active">✓ Aktif filtre: {" · ".join(etiketler)} '
                f'— <b>{len(dfd):,} haber</b></div>',
                unsafe_allow_html=True,
            )
    else:
        # Filtre uygulanmamışsa son 12 periyot göster
        dfd = df_periyot_filtrele(df, granularity, secili_periyotlar)
        st.markdown(
            f'<div class="filter-active" style="background:rgba(148,163,184,0.1);'
            f'color:#94A3B8;border-color:rgba(148,163,184,0.3)">'
            f'Filtre uygulanmadı — varsayılan dönem gösteriliyor · <b>{len(dfd):,} haber</b></div>',
            unsafe_allow_html=True,
        )

    if dfd.empty:
        st.warning("Filtre sonucu boş — farklı bir dönem veya ülke seçin.")
        st.stop()

    # ── Metrikler ────────────────────────────────────────────────────
    peak       = dfd.groupby("date_only").size()
    peak_date  = str(peak.idxmax()) if len(peak) else "—"
    peak_cnt   = int(peak.max()) if len(peak) else 0
    span_days  = max((dfd["date_only"].max() - dfd["date_only"].min()).days + 1, 1)
    daily_avg  = len(dfd) / span_days
    top_ulke   = dfd["country_name"].value_counts()
    top_src    = dfd["source"].value_counts()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub in [
        (c1, "Toplam Haber",           f"{len(dfd):,}",
         f"Günlük ort. {daily_avg:.1f}"),
        (c2, "Başat Ülke",             top_ulke.index[0] if len(top_ulke) else "—",
         f"{int(top_ulke.iloc[0]) if len(top_ulke) else 0:,} haber"),
        (c3, "En Çok Haber Veren Kaynak", top_src.index[0] if len(top_src) else "—",
         f"{int(top_src.iloc[0]) if len(top_src) else 0:,} haber"),
        (c4, "En Yoğun Gün",           peak_date, f"{peak_cnt} haber"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value" style="font-size:1.1rem">{val}</div>'
                f'<div class="metric-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Haber akışı ──
    st.markdown('<div class="section-title">Haber Akışı</div>', unsafe_allow_html=True)
    ts = pick_granularity(dfd, granularity)
    fig_t = go.Figure(go.Bar(x=ts["label"], y=ts["count"],
                             marker_color=AMBER, marker_line_width=0))
    # Yıllık modda integer tick zorla — 2022.5 gibi ara değerleri önler
    xaxis_extra = (
        dict(tickmode="linear", dtick=1, tickformat="d")
        if granularity == "Yıllık" else {}
    )
    fig_t.update_layout(**PLOTLY_LAYOUT, yaxis_title="Haber Sayısı", height=280)
    fig_t.update_xaxes(**AXIS_STYLE, tickangle=-45, **xaxis_extra)
    fig_t.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_t, use_container_width=True)

    # ── Ülke dağılımı ──
    st.markdown('<div class="section-title">Ülkelere Göre Dağılım</div>', unsafe_allow_html=True)
    ulke_counts = dfd["country_name"].value_counts().head(10)
    fig_u = go.Figure(go.Bar(
        x=ulke_counts.values, y=ulke_counts.index, orientation="h",
        marker=dict(color=COLORS[:len(ulke_counts)]),
        text=ulke_counts.values, textposition="outside",
        textfont=dict(color="#CBD5E1"),
    ))
    fig_u.update_layout(**PLOTLY_LAYOUT, margin=dict(t=10, b=10, l=130, r=60), height=320)
    fig_u.update_xaxes(**AXIS_STYLE)
    fig_u.update_yaxes(**AXIS_STYLE, autorange="reversed")
    st.plotly_chart(fig_u, use_container_width=True)

    # ── Ülke bazlı zaman serisi ──
    st.markdown('<div class="section-title">Ülke Bazlı Haber Akışı</div>', unsafe_allow_html=True)
    ulke_ts = dfd.groupby(["ym", "country_name"]).size().reset_index(name="count")
    fig_uts = go.Figure()
    for i, ulke in enumerate(ULKELER):
        sub = ulke_ts[ulke_ts["country_name"] == ulke]
        if not sub.empty:
            fig_uts.add_trace(go.Scatter(
                x=sub["ym"], y=sub["count"],
                mode="lines+markers", name=ulke,
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                marker=dict(size=4),
            ))
    fig_uts.update_layout(**PLOTLY_LAYOUT, height=320,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      font=dict(color="#CBD5E1")))
    fig_uts.update_xaxes(**AXIS_STYLE, tickangle=-30)
    fig_uts.update_yaxes(**AXIS_STYLE, title="Haber Sayısı")
    st.plotly_chart(fig_uts, use_container_width=True)

    # ── Isı haritası ──
    st.markdown('<div class="section-title">Haber Yoğunluğu Isı Haritası</div>',
                unsafe_allow_html=True)
    dash_heat_gran = st.radio(
        "Granülerite", ["Aylık", "Haftalık"], horizontal=True, key="dash_heat_gran"
    )
    render_isi_haritasi(dfd, dash_heat_gran, STOPWORDS, key_prefix="dash_heat")

    # ── Kaynak sıralaması ──
    st.markdown('<div class="section-title">Kaynak Sıralaması</div>', unsafe_allow_html=True)
    src15 = dfd["source"].value_counts().head(15).reset_index()
    src15.columns = ["Kaynak", "Haber"]
    fig_s = go.Figure(go.Bar(
        x=src15["Haber"], y=src15["Kaynak"], orientation="h",
        marker=dict(color=src15["Haber"],
                    colorscale=[[0, "#1E3A5F"], [0.5, TEAL], [1.0, AMBER]],
                    showscale=False),
        text=src15["Haber"], textposition="outside",
        textfont=dict(color="#CBD5E1"),
    ))
    fig_s.update_layout(**PLOTLY_LAYOUT, margin=dict(t=10, b=10, l=180, r=60), height=460)
    fig_s.update_xaxes(**AXIS_STYLE)
    fig_s.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_s, use_container_width=True)

    # ── Kaynak çeşitliliği ve haber uzunluğu yan yana ──
    row2_l, row2_r = st.columns(2)
    with row2_l:
        st.markdown('<div class="section-title">Ülke Başına Kaynak Çeşitliliği</div>',
                    unsafe_allow_html=True)
        cesitlilik = dfd.groupby("country_name")["source"].nunique().sort_values(ascending=False)
        fig_cev = go.Figure(go.Bar(
            x=cesitlilik.index, y=cesitlilik.values,
            marker=dict(color=COLORS[:len(cesitlilik)]),
            text=cesitlilik.values, textposition="outside",
            textfont=dict(color="#CBD5E1"),
        ))
        fig_cev.update_layout(**PLOTLY_LAYOUT, yaxis_title="Benzersiz Kaynak", height=300,
                              margin=dict(t=10, b=10, l=20, r=20))
        fig_cev.update_xaxes(**AXIS_STYLE)
        fig_cev.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_cev, use_container_width=True)

    with row2_r:
        st.markdown('<div class="section-title">Ortalama Haber Uzunluğu (Kelime)</div>',
                    unsafe_allow_html=True)
        uzunluk = dfd.groupby("country_name")["word_count"].mean().sort_values(ascending=False).round(0)
        fig_uz = go.Figure(go.Bar(
            x=uzunluk.index, y=uzunluk.values,
            marker=dict(color=COLORS[:len(uzunluk)]),
            text=[f"{v:.0f}" for v in uzunluk.values], textposition="outside",
            textfont=dict(color="#CBD5E1"),
        ))
        fig_uz.update_layout(**PLOTLY_LAYOUT, yaxis_title="Ort. Kelime", height=300,
                             margin=dict(t=10, b=10, l=20, r=20))
        fig_uz.update_xaxes(**AXIS_STYLE)
        fig_uz.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_uz, use_container_width=True)

    # ── Terimler (unigram / bigram / trigram) ──
    st.markdown('<div class="section-title">En Sık Geçen Terimler</div>', unsafe_allow_html=True)
    gram_secim = st.radio(
        "Gram türü", ["Tek kelime", "İki kelime", "Üç kelime"],
        horizontal=True, key="dash_gram"
    )
    ngrams = top_terms_by_gram(dfd, STOPWORDS, n=20)
    gram_map = {"Tek kelime": "uni", "İki kelime": "bi", "Üç kelime": "tri"}
    gram_renkler = {"uni": TEAL, "bi": AMBER, "tri": VIOLET}
    secilen = gram_map[gram_secim]
    ngram_grafigi(
        ngrams[secilen],
        baslik=gram_secim,
        renk=gram_renkler[secilen],
        key=f"dash_ngram_{secilen}",
        yukseklik=500,
    )


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — KARŞILAŞTIRMA
# ═══════════════════════════════════════════════════════════════════

with tab_compare:
    st.markdown("### ⚖️ İki Dönem / Grup Karşılaştırması")

    col_a, col_b = st.columns(2)
    tum_periyotlar_comp = periyot_secenekleri(df, "Aylık")  # karşılaştırma aylık başlar

    with col_a:
        st.markdown('<div class="compare-header-a">🔵 Grup A</div>', unsafe_allow_html=True)
        a_gran = st.selectbox("Zaman birimi", ["Aylık", "Haftalık", "Yıllık", "Günlük"], key="a_gran")
        a_pers = periyot_secenekleri(df, a_gran)
        a_per  = st.multiselect("Dönem", a_pers,
                                 default=a_pers[-6:] if len(a_pers) >= 6 else a_pers,
                                 key="a_per")
        a_ulk  = st.multiselect("Ülke", ULKELER, key="a_ulk")
        a_src  = st.multiselect("Kaynak", all_sources, key="a_src")
        a_lbl  = st.text_input("Etiket", value="Grup A", key="a_lbl")

    with col_b:
        st.markdown('<div class="compare-header-b">🟡 Grup B</div>', unsafe_allow_html=True)
        b_gran = st.selectbox("Zaman birimi", ["Aylık", "Haftalık", "Yıllık", "Günlük"], key="b_gran")
        b_pers = periyot_secenekleri(df, b_gran)
        b_per  = st.multiselect("Dönem", b_pers,
                                 default=b_pers[-12:-6] if len(b_pers) >= 12 else b_pers[:6],
                                 key="b_per")
        b_ulk  = st.multiselect("Ülke", ULKELER, key="b_ulk")
        b_src  = st.multiselect("Kaynak", all_sources, key="b_src")
        b_lbl  = st.text_input("Etiket", value="Grup B", key="b_lbl")
    # Radio butonlar her zaman görünür — session_state'e bağlı
    comp_gram_secim = st.radio(
        "Terim gram türü", ["Tek kelime", "İki kelime", "Üç kelime"],
        horizontal=True, key="comp_gram"
    )
    comp_heat_gran = st.radio(
        "Isı haritası granüleritesi", ["Aylık", "Haftalık"],
        horizontal=True, key="comp_heat_gran"
    )
    c_ton = st.checkbox("Ton analizi dahil et (LLM)", key="c_ton")
    c_run = st.button("⚖️ Karşılaştır", type="primary")

    # Karşılaştır butonuna basınca hesapla ve session_state'e kaydet
    if c_run:
        dfa = df_periyot_filtrele(df, a_gran, a_per)
        dfb = df_periyot_filtrele(df, b_gran, b_per)
        if a_ulk: dfa = dfa[dfa["country_name"].isin(a_ulk)]
        if a_src: dfa = dfa[dfa["source"].isin(a_src)]
        if b_ulk: dfb = dfb[dfb["country_name"].isin(b_ulk)]
        if b_src: dfb = dfb[dfb["source"].isin(b_src)]

        if len(dfa) == 0 or len(dfb) == 0:
            st.warning("Gruplardan biri boş — dönem seçimini genişlet.")
            st.session_state.pop("comp_sonuc", None)
        else:
            terms_a_tmp = top_terms(dfa, 50, STOPWORDS)
            terms_b_tmp = top_terms(dfb, 50, STOPWORDS)
            d_inc, d_dec = term_diff(terms_a_tmp, terms_b_tmp)
            st.session_state["comp_sonuc"] = {
                "dfa":      dfa,
                "dfb":      dfb,
                "a_gran":   a_gran,
                "b_gran":   b_gran,
                "a_lbl":    a_lbl,
                "b_lbl":    b_lbl,
                "c_ton":    c_ton,
                "ngrams_a": top_terms_by_gram(dfa, STOPWORDS, n=20),
                "ngrams_b": top_terms_by_gram(dfb, STOPWORDS, n=20),
                "diff_inc": d_inc,
                "diff_dec": d_dec,
            }

    # Sonuçları session_state'ten çiz — radio değişse de kaybolmaz
    if "comp_sonuc" in st.session_state:
        cs     = st.session_state["comp_sonuc"]
        dfa    = cs["dfa"];     dfb    = cs["dfb"]
        a_gran = cs["a_gran"];  b_gran = cs["b_gran"]
        a_lbl  = cs["a_lbl"];  b_lbl  = cs["b_lbl"]
        c_ton  = cs["c_ton"]
        ngrams_a = cs["ngrams_a"]; ngrams_b = cs["ngrams_b"]
        diff_inc = cs["diff_inc"]; diff_dec = cs["diff_dec"]

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        delta = len(dfa) - len(dfb)
        tu_a  = dfa["country_name"].value_counts().index[0] if len(dfa) else "—"
        tu_b  = dfb["country_name"].value_counts().index[0] if len(dfb) else "—"
        src_a = dfa["source"].value_counts().index[0] if len(dfa) else "—"
        src_b = dfb["source"].value_counts().index[0] if len(dfb) else "—"

        with m1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Haber Sayısı</div>'
                f'<div class="metric-value" style="font-size:1.1rem">'
                f'<span style="color:{SKY}">{len(dfa):,}</span> vs '
                f'<span style="color:{AMBER}">{len(dfb):,}</span></div>'
                f'<div class="metric-sub">Δ {"+" if delta>0 else ""}{delta}</div></div>',
                unsafe_allow_html=True)
        with m2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Başat Ülke</div>'
                f'<div class="metric-value" style="font-size:0.95rem">'
                f'<span style="color:{SKY}">{tu_a}</span><br>'
                f'<span style="color:{AMBER}">{tu_b}</span></div></div>',
                unsafe_allow_html=True)
        with m3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Öne Çıkan Kaynak</div>'
                f'<div class="metric-value" style="font-size:0.8rem">'
                f'<span style="color:{SKY}">{src_a}</span><br>'
                f'<span style="color:{AMBER}">{src_b}</span></div></div>',
                unsafe_allow_html=True)
        with m4:
            ua, ub = dfa["source"].nunique(), dfb["source"].nunique()
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Kaynak Çeşitliliği</div>'
                f'<div class="metric-value" style="font-size:1.1rem">'
                f'<span style="color:{SKY}">{ua}</span> vs '
                f'<span style="color:{AMBER}">{ub}</span></div>'
                f'<div class="metric-sub">farklı yayın organı</div></div>',
                unsafe_allow_html=True)

        st.markdown("---")

        # Haber akışı
        st.markdown('<div class="section-title">Haber Akışı Karşılaştırması</div>', unsafe_allow_html=True)
        tsa = pick_granularity(dfa, a_gran)
        tsb = pick_granularity(dfb, b_gran)
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=tsa["label"], y=tsa["count"], mode="lines+markers",
                                    name=a_lbl, line=dict(color=SKY, width=2), marker=dict(color=SKY, size=5)))
        fig_ts.add_trace(go.Scatter(x=tsb["label"], y=tsb["count"], mode="lines+markers",
                                    name=b_lbl, line=dict(color=AMBER, width=2), marker=dict(color=AMBER, size=5)))
        fig_ts.update_layout(**PLOTLY_LAYOUT, yaxis_title="Haber Sayısı", height=300,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#CBD5E1")))
        fig_ts.update_xaxes(**AXIS_STYLE)
        fig_ts.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_ts, use_container_width=True)

        # Kaynak karşılaştırması
        st.markdown('<div class="section-title">Kaynak Dağılımı Karşılaştırması</div>', unsafe_allow_html=True)
        top_srcs = list(dict.fromkeys(
            list(dfa["source"].value_counts().head(8).index) +
            list(dfb["source"].value_counts().head(8).index)
        ))
        ta_, tb_ = len(dfa) or 1, len(dfb) or 1
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Bar(name=a_lbl, x=top_srcs,
            y=[dfa[dfa["source"]==s].shape[0]/ta_*100 for s in top_srcs], marker_color=SKY))
        fig_sc.add_trace(go.Bar(name=b_lbl, x=top_srcs,
            y=[dfb[dfb["source"]==s].shape[0]/tb_*100 for s in top_srcs], marker_color=AMBER))
        fig_sc.update_layout(**PLOTLY_LAYOUT, barmode="group", yaxis_title="% (normalize)", height=350,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#CBD5E1")))
        fig_sc.update_xaxes(**AXIS_STYLE, tickangle=-40)
        fig_sc.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_sc, use_container_width=True)

        # Terim farklılıkları — comp_gram_secim radio'dan gelir
        st.markdown('<div class="section-title">Terim Farklılıkları</div>', unsafe_allow_html=True)
        comp_gram_map = {"Tek kelime": "uni", "İki kelime": "bi", "Üç kelime": "tri"}
        comp_gram_key = comp_gram_map[comp_gram_secim]
        diff_inc_g, diff_dec_g = term_diff(ngrams_a[comp_gram_key], ngrams_b[comp_gram_key])

        tl, tr = st.columns(2)
        with tl:
            st.markdown(f'<div class="compare-header-a" style="font-size:0.85rem">{a_lbl} — öne çıkan</div>',
                        unsafe_allow_html=True)
            if diff_inc_g:
                wi, _, _, si = zip(*diff_inc_g[:15])
                fig_i = go.Figure(go.Bar(x=list(si), y=list(wi), orientation="h", marker_color=SKY,
                    text=[f"{v:.4f}" for v in si], textposition="outside", textfont=dict(color="#CBD5E1")))
                fig_i.update_layout(**PLOTLY_LAYOUT,
                    margin=dict(t=10, b=10, l=max(80, max(len(w) for w in wi)*7), r=60), height=420)
                fig_i.update_xaxes(**AXIS_STYLE)
                fig_i.update_yaxes(**AXIS_STYLE, autorange="reversed")
                st.plotly_chart(fig_i, use_container_width=True, key="comp_diff_a")
            else:
                st.caption("Yeterli veri yok.")
        with tr:
            st.markdown(f'<div class="compare-header-b" style="font-size:0.85rem">{b_lbl} — öne çıkan</div>',
                        unsafe_allow_html=True)
            if diff_dec_g:
                wd, _, _, sd = zip(*diff_dec_g[:15])
                fig_d = go.Figure(go.Bar(x=[abs(v) for v in sd], y=list(wd), orientation="h", marker_color=AMBER,
                    text=[f"{abs(v):.4f}" for v in sd], textposition="outside", textfont=dict(color="#CBD5E1")))
                fig_d.update_layout(**PLOTLY_LAYOUT,
                    margin=dict(t=10, b=10, l=max(80, max(len(w) for w in wd)*7), r=60), height=420)
                fig_d.update_xaxes(**AXIS_STYLE)
                fig_d.update_yaxes(**AXIS_STYLE, autorange="reversed")
                st.plotly_chart(fig_d, use_container_width=True, key="comp_diff_b")
            else:
                st.caption("Yeterli veri yok.")

        # Isı haritaları — comp_heat_gran radio'dan gelir
        st.markdown("---")
        st.markdown('<div class="section-title">Haber Yoğunluğu Isı Haritası</div>', unsafe_allow_html=True)
        ch_l, ch_r = st.columns(2)
        with ch_l:
            st.markdown(f'<div class="compare-header-a" style="font-size:0.85rem">{a_lbl}</div>',
                        unsafe_allow_html=True)
            render_isi_haritasi(dfa, comp_heat_gran, STOPWORDS, key_prefix="comp_heat_a")
        with ch_r:
            st.markdown(f'<div class="compare-header-b" style="font-size:0.85rem">{b_lbl}</div>',
                        unsafe_allow_html=True)
            render_isi_haritasi(dfb, comp_heat_gran, STOPWORDS, key_prefix="comp_heat_b")

        # Ton analizi (opsiyonel)
        if c_ton:
            st.markdown("---")
            st.markdown('<div class="section-title">Ton Analizi</div>', unsafe_allow_html=True)
            client, model = get_client(
                st.session_state.get("llm_tipi", "openai"),
                st.session_state.get("api_key", ""),
                st.session_state.get("ollama_url", ""),
            )
            with st.spinner(f"{a_lbl} ton analizi..."):
                ton_a = llm_ton_analizi(client, model, dfa, a_lbl)
            with st.spinner(f"{b_lbl} ton analizi..."):
                ton_b = llm_ton_analizi(client, model, dfb, b_lbl)

            ton_keys   = ["destekleyici", "nötr", "eleştirel", "kriz_odakli", "diplomatik"]
            ton_labels = ["Destekleyici", "Nötr", "Eleştirel", "Kriz Odaklı", "Diplomatik"]
            ton_colors = [LIME, SLATE, ROSE, VIOLET, TEAL]

            tona_c, tonb_c = st.columns(2)
            for col_t, ton, lbl in [(tona_c, ton_a, a_lbl), (tonb_c, ton_b, b_lbl)]:
                with col_t:
                    fig_ton = go.Figure(go.Pie(
                        labels=ton_labels, values=[ton.get(k, 0) for k in ton_keys],
                        marker=dict(colors=ton_colors), textinfo="label+percent",
                        textfont=dict(color="#F8FAFC", size=10),
                    ))
                    fig_ton.update_layout(**PLOTLY_LAYOUT,
                        title=dict(text=lbl, font=dict(size=13, color="#F8FAFC")),
                        showlegend=False, height=320, margin=dict(t=40, b=10, l=10, r=10))
                    st.plotly_chart(fig_ton, use_container_width=True)
                    st.caption(ton.get("özet", "—"))

            st.markdown("---")
            st.markdown('<div class="section-title">Karşılaştırmalı Yorum</div>', unsafe_allow_html=True)
            with st.spinner("LLM analiz yapıyor..."):
                stats_a = {"count": len(dfa), "ton": ton_a}
                stats_b = {"count": len(dfb), "ton": ton_b}
                yorum = llm_karsilastirma_yorumu(
                    client, model, stats_a, stats_b, a_lbl, b_lbl, (diff_inc, diff_dec))
            st.markdown(yorum)

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — RAPOR ÜRETİMİ
# ═══════════════════════════════════════════════════════════════════

with tab_rapor:
    st.markdown("### 📋 Analitik Rapor Üretimi")

    r_col1, r_col2 = st.columns([1, 1])

    with r_col1:
        r_ulkeler = st.multiselect(
            "Ülke(ler)", ULKELER,
            default=["Mısır"] if "Mısır" in ULKELER else [],
            key="r_ulke",
        )
        r_gran = st.selectbox(
            "Zaman birimi", ["Aylık", "Haftalık", "Yıllık", "Günlük"], key="r_gran"
        )
        r_pers_all = periyot_secenekleri(
            df[df["country_name"].isin(r_ulkeler)] if r_ulkeler else df, r_gran
        )
        r_per = st.multiselect(
            "Dönem", r_pers_all,
            default=r_pers_all[-3:] if len(r_pers_all) >= 3 else r_pers_all,
            key="r_per",
        )
        r_sources = st.multiselect("Kaynak filtresi (boş = tümü)", all_sources, key="r_src")

    with r_col2:
        st.markdown("**Politika Notu Seçimi**")
        policy_notes = st.session_state.get("policy_notes", {})
        secili_notlar: dict[str, str] = {}

        if r_ulkeler:
            for ulke in r_ulkeler:
                ulke_notlari = policy_notes.get(ulke, {})
                if ulke_notlari:
                    secenekler = list(ulke_notlari.keys())
                    secim = st.multiselect(
                        f"{ulke} — Notlar", secenekler,
                        default=secenekler, key=f"pnot_{ulke}",
                    )
                    for s in secim:
                        secili_notlar[f"{ulke}/{s}"] = ulke_notlari[s]
                else:
                    st.caption(f"{ulke}: `{POLICY_DIR}/{ulke.lower()}/` klasöründe PDF bulunamadı")

        if secili_notlar:
            st.success(f"{len(secili_notlar)} politika notu seçili")
            for k in secili_notlar:
                st.markdown(f'<div class="policy-box">📄 {k}</div>', unsafe_allow_html=True)
        else:
            st.warning("Rapor için en az bir politika notu gerekli")

    # Haber filtrele & önizle
    st.markdown("---")
    r_df = df_periyot_filtrele(df, r_gran, r_per)
    if r_ulkeler:
        r_df = r_df[r_df["country_name"].isin(r_ulkeler)]
    if r_sources:
        r_df = r_df[r_df["source"].isin(r_sources)]

    if not r_df.empty:
        st.info(
            f"**{len(r_df)} haber** seçildi | "
            f"Dönem: {r_df['date_only'].min()} → {r_df['date_only'].max()} | "
            f"{r_df['source'].nunique()} kaynak"
        )
        with st.expander("Haberleri önizle", expanded=True):
            st.dataframe(
                r_df[["date_only", "country_name", "source", "title", "word_count"]]
                .sort_values("date_only")
                .rename(columns={"date_only": "tarih"}),
                use_container_width=True, height=250,
            )
    else:
        st.warning("Seçilen filtrelere uyan haber bulunamadı.")

    # LLM haber limiti kontrolü
    llm_df = r_df.copy()
    if len(llm_df) > MAX_HABER_LLM:
        # Temsili haberler seçimi — rastgele değil, skor bazlı
        skorlu = temsili_haberler(llm_df, STOPWORDS, MAX_HABER_LLM)
        secili_hash = {t["url"] for t in skorlu}
        llm_df = llm_df[llm_df["url"].isin(secili_hash)].head(MAX_HABER_LLM)
        st.warning(
            f"⚠️ Seçilen {len(r_df):,} haberden en temsili **{MAX_HABER_LLM}** tanesi "
            f"LLM'e gönderilecek (context window limiti). "
            f"Seçim dönemin öne çıkan terimleriyle eşleşen haberler üzerinden yapılmıştır."
        )

    can_generate = (
        not r_df.empty
        and bool(secili_notlar)
        and (
            st.session_state.get("llm_tipi") == "ollama"
            or bool(st.session_state.get("api_key") or os.environ.get("OPENAI_API_KEY"))
        )
    )
    rapor_baslat = st.button("🚀 Rapor Üret", type="primary", disabled=not can_generate)

    if rapor_baslat:
        client, model = get_client(
            st.session_state.get("llm_tipi", "openai"),
            st.session_state.get("api_key", ""),
            st.session_state.get("ollama_url", ""),
        )

        ulke_str     = ", ".join(r_ulkeler) if r_ulkeler else "Seçili ülkeler"
        tarih_str    = f"{r_df['date_only'].min()} – {r_df['date_only'].max()}"
        kaynaklar_st = kaynak_dagılımı(r_df)
        haberler_str = haberleri_formatla(llm_df)
        politika_str = "\n\n---\n\n".join(
            [f"[{k}]\n{v}" for k, v in secili_notlar.items()]
        )

        st.markdown("---")
        st.subheader("📊 Aşama 1: Tematik Haber Haritası")
        ph1 = st.empty()
        with st.spinner("Haberler analiz ediliyor..."):
            s1 = llm_call(client, model, SISTEM_PROMPT, PROMPT_1.format(
                ulke=ulke_str, tarih_araligi=tarih_str,
                haber_sayisi=len(llm_df), haberler=haberler_str,
            ))
            haber_haritasi = stream_write(ph1, s1)

        st.subheader("🔍 Aşama 2: Analitik Değerlendirme")
        ph2 = st.empty()
        with st.spinner("Politika notu ile karşılaştırılıyor..."):
            s2 = llm_call(client, model, SISTEM_PROMPT, PROMPT_2.format(
                ulke=ulke_str, haber_haritasi=haber_haritasi, politika_notu=politika_str,
            ))
            analitik = stream_write(ph2, s2)

        st.subheader("🔮 Aşama 3: Olası Seyirler ve Politika Önerileri")
        ph3 = st.empty()
        with st.spinner("Seyirler ve öneriler oluşturuluyor..."):
            s4 = llm_call(client, model, SISTEM_PROMPT, PROMPT_4.format(
                ulke=ulke_str,
                onceki_analizler=f"Tematik Harita:\n{haber_haritasi}\n\nAnalitik:\n{analitik}",
            ))
            seyirler = stream_write(ph3, s4)

        st.markdown("---")
        st.subheader("📋 Nihai Rapor")
        ph5 = st.empty()
        with st.spinner("Nihai rapor derleniyor..."):
            s5 = llm_call(client, model, SISTEM_PROMPT, PROMPT_5.format(
                haber_haritasi=haber_haritasi, analitik=analitik, seyirler=seyirler,
                ulke=ulke_str, tarih_araligi=tarih_str,
                haber_sayisi=len(llm_df), kaynaklar=kaynaklar_st,
            ))
            nihai_rapor = stream_write(ph5, s5)

        # Tüm aşamaları tek belgede birleştir
        tam_belge = f"""# {ulke_str} Analiz Raporu | {tarih_str}

---

## Aşama 1: Tematik Haber Haritası

{haber_haritasi}

---

## Aşama 2: Analitik Değerlendirme

{analitik}

---

## Aşama 3: Olası Seyirler ve Politika Önerileri

{seyirler}

---

## Nihai Rapor

{nihai_rapor}
"""

        # İndirme
        st.markdown("---")
        dosya_tabanı = (
            f"rapor_{ulke_str.replace(', ', '_').lower()}_"
            f"{tarih_str.replace(' ', '').replace('–', '_')}"
        )
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Markdown olarak indir (.md)",
                data=tam_belge,
                file_name=f"{dosya_tabanı}.md",
                mime="text/markdown",
            )
        with dl2:
            with st.spinner("PDF oluşturuluyor..."):
                try:
                    pdf_bytes = markdown_to_pdf(tam_belge)
                    st.download_button(
                        "📄 PDF olarak indir",
                        data=pdf_bytes,
                        file_name=f"{dosya_tabanı}.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.error(f"PDF oluşturulamadı: {e}")
        st.success("✅ Rapor tamamlandı.")
