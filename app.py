"""
LOT Polish Airlines – System Predykcji Nastrojów Pasażerów
==========================================================
Uruchomienie:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from model import SentimentModel, CrisisScorer
from data_generator import generate_sample_data

# ─── KONFIGURACJA STRONY ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="LOT Sentiment Monitor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap');

    [data-testid="stAppViewContainer"] { background: #07090F; }
    [data-testid="stSidebar"] { background: #0D1117; border-right: 1px solid #1A2030; }
    [data-testid="metric-container"] {
        background: #1A2030;
        border: 1px solid #212840;
        border-radius: 12px;
        padding: 16px !important;
    }
    .metric-label { color: #6B7A99 !important; font-size: 11px !important; }
    h1, h2, h3 { color: #E8EDF5 !important; }
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    .crisis-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
    }
    .high { background: rgba(239,68,68,0.15); color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }
    .mid  { background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3); }
    .low  { background: rgba(34,197,94,0.15);  color: #22C55E; border: 1px solid rgba(34,197,94,0.3); }

    div[data-testid="stTabs"] button {
        color: #6B7A99;
        font-weight: 600;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #C8A951;
        border-bottom-color: #C8A951;
    }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ LOT Monitor")
    st.markdown("**System Predykcji Nastrojów**")
    st.divider()

    n_opinions = st.slider("Liczba opinii do analizy", 50, 500, 200, 50)
    date_range = st.slider("Zakres dni wstecz", 7, 90, 30)
    crisis_threshold = st.slider("Próg alertu Crisis Score", 50, 90, 70)

    st.divider()
    st.markdown("**Filtruj kategorie:**")
    show_delay     = st.checkbox("✈️ Opóźnienia", True)
    show_baggage   = st.checkbox("🧳 Bagaż", True)
    show_service   = st.checkbox("👤 Obsługa", True)
    show_comms     = st.checkbox("📞 Komunikacja", True)
    show_comp      = st.checkbox("💶 Odszkodowania", True)

    st.divider()
    if st.button("🔄 Odśwież dane", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        "<div style='color:#6B7A99;font-size:11px;margin-top:16px;'>"
        "Model: XLM-RoBERTa · HerBERT<br>"
        "Dane: syntetyczne (PL)<br>"
        "Wersja: 1.0.0</div>",
        unsafe_allow_html=True
    )

# ─── ŁADOWANIE DANYCH ─────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data(n, days):
    df = generate_sample_data(n, days)
    model = SentimentModel()
    scorer = CrisisScorer(threshold=70)
    df = model.predict(df)
    df = scorer.score(df)
    return df

df_all = load_data(n_opinions, date_range)

# Filtr kategorii
active_cats = []
if show_delay:   active_cats.append("opóźnienie")
if show_baggage: active_cats.append("bagaż")
if show_service: active_cats.append("obsługa")
if show_comms:   active_cats.append("komunikacja")
if show_comp:    active_cats.append("odszkodowanie")

if active_cats:
    df = df_all[df_all["category"].isin(active_cats)].copy()
else:
    df = df_all.copy()

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔬 Analizator opinii", "📋 Tabela danych"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📡 Monitor Nastrojów Pasażerów LOT")

    # ── METRYKI ────────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    total = len(df)
    neg_pct = round(len(df[df.sentiment == "negatywny"]) / total * 100, 1) if total else 0
    avg_crisis = round(df.crisis_score.mean(), 1) if total else 0
    alerts_high = len(df[df.crisis_score >= crisis_threshold])
    pos_pct = round(len(df[df.sentiment == "pozytywny"]) / total * 100, 1) if total else 0

    col1.metric("📝 Łączna liczba opinii", total)
    col2.metric("😡 Negatywne opinie", f"{neg_pct}%", delta=f"+{round(neg_pct-54,1)}% vs poprz. mies.", delta_color="inverse")
    col3.metric("⚡ Śr. Crisis Score", f"{avg_crisis}/100", delta=f"+{round(avg_crisis-55,1)} vs poprz. mies.", delta_color="inverse")
    col4.metric(f"🔴 Alerty (próg {crisis_threshold})", alerts_high, delta=f"+{max(0, alerts_high-15)} dziś", delta_color="inverse")
    col5.metric("😊 Pozytywne opinie", f"{pos_pct}%", delta=f"{round(pos_pct-26,1)}% vs poprz. mies.", delta_color="normal")

    st.divider()

    # ── ROW 2: TREND + DONUT ──────────────────────────────────────────────────
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown("#### Trend sentymentu w czasie")
        daily = df.groupby(["date", "sentiment"]).size().reset_index(name="count")
        daily_pivot = daily.pivot(index="date", columns="sentiment", values="count").fillna(0).reset_index()
        for col in ["negatywny", "neutralny", "pozytywny"]:
            if col not in daily_pivot.columns:
                daily_pivot[col] = 0

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=daily_pivot["date"], y=daily_pivot["negatywny"],
            name="Negatywne", fill="tozeroy",
            line=dict(color="#EF4444", width=2),
            fillcolor="rgba(239,68,68,0.15)"
        ))
        fig_trend.add_trace(go.Scatter(
            x=daily_pivot["date"], y=daily_pivot["pozytywny"],
            name="Pozytywne", fill="tozeroy",
            line=dict(color="#22C55E", width=2),
            fillcolor="rgba(34,197,94,0.15)"
        ))
        fig_trend.add_trace(go.Scatter(
            x=daily_pivot["date"], y=daily_pivot["neutralny"],
            name="Neutralne", line=dict(color="#F59E0B", width=1.5, dash="dot")
        ))
        fig_trend.update_layout(
            paper_bgcolor="#1A2030", plot_bgcolor="#1A2030",
            font=dict(color="#94A3B8", size=12),
            legend=dict(bgcolor="#212840", bordercolor="#2D3748"),
            margin=dict(l=0, r=0, t=0, b=0), height=280,
            xaxis=dict(gridcolor="#212840"),
            yaxis=dict(gridcolor="#212840"),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_r:
        st.markdown("#### Rozkład sentymentu")
        sent_counts = df["sentiment"].value_counts()
        colors = {"negatywny": "#EF4444", "neutralny": "#F59E0B", "pozytywny": "#22C55E"}
        fig_pie = go.Figure(go.Pie(
            labels=sent_counts.index,
            values=sent_counts.values,
            hole=0.6,
            marker_colors=[colors.get(l, "#6B7A99") for l in sent_counts.index],
            textinfo="percent",
            textfont=dict(color="white", size=13),
        ))
        fig_pie.update_layout(
            paper_bgcolor="#1A2030", plot_bgcolor="#1A2030",
            font=dict(color="#94A3B8"),
            legend=dict(bgcolor="#212840"),
            margin=dict(l=0, r=0, t=0, b=0), height=280,
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── ROW 3: CATEGORIES + CRISIS SCORE ──────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Kategorie problemów (negatywne)")
        neg_df = df[df.sentiment == "negatywny"]
        cat_counts = neg_df["category"].value_counts().reset_index()
        cat_counts.columns = ["Kategoria", "Liczba"]
        cat_emoji = {
            "opóźnienie": "✈️ Opóźnienia",
            "bagaż": "🧳 Bagaż",
            "obsługa": "👤 Obsługa",
            "komunikacja": "📞 Komunikacja",
            "odszkodowanie": "💶 Odszkodowania",
            "inne": "❓ Inne",
        }
        cat_counts["Kategoria"] = cat_counts["Kategoria"].map(lambda x: cat_emoji.get(x, x))
        fig_bar = px.bar(
            cat_counts, x="Liczba", y="Kategoria", orientation="h",
            color="Liczba", color_continuous_scale=["#F59E0B", "#EF4444"],
        )
        fig_bar.update_layout(
            paper_bgcolor="#1A2030", plot_bgcolor="#1A2030",
            font=dict(color="#94A3B8"),
            margin=dict(l=0, r=0, t=0, b=0), height=280,
            coloraxis_showscale=False,
            yaxis=dict(gridcolor="#212840"),
            xaxis=dict(gridcolor="#212840"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.markdown("#### Rozkład Crisis Score")
        fig_hist = px.histogram(
            df, x="crisis_score", nbins=20,
            color_discrete_sequence=["#C8A951"],
        )
        fig_hist.add_vline(
            x=crisis_threshold, line_dash="dash",
            line_color="#EF4444", annotation_text=f"Próg alertu ({crisis_threshold})",
            annotation_font_color="#EF4444",
        )
        fig_hist.update_layout(
            paper_bgcolor="#1A2030", plot_bgcolor="#1A2030",
            font=dict(color="#94A3B8"),
            margin=dict(l=0, r=0, t=0, b=0), height=280,
            xaxis=dict(gridcolor="#212840"),
            yaxis=dict(gridcolor="#212840", title="Liczba opinii"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── ALERTY ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"### 🚨 Opinie wymagające reakcji (Crisis Score ≥ {crisis_threshold})")

    alerts_df = df[df.crisis_score >= crisis_threshold].sort_values("crisis_score", ascending=False).head(10)

    if alerts_df.empty:
        st.success("✅ Brak alertów przy aktualnym progu.")
    else:
        for _, row in alerts_df.iterrows():
            score = row["crisis_score"]
            badge_cls = "high" if score >= 80 else "mid"
            with st.container():
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"""
                    <div style='background:#1A2030;border:1px solid #2D3748;border-radius:10px;padding:14px;margin-bottom:10px;border-left:3px solid {"#EF4444" if score>=80 else "#F59E0B"}'>
                        <div style='color:#E8EDF5;font-size:14px;line-height:1.6;margin-bottom:8px;'>"{row['text']}"</div>
                        <div style='display:flex;gap:8px;flex-wrap:wrap;'>
                            <span class='crisis-badge {badge_cls}'>Crisis: {score}</span>
                            <span style='background:#212840;border:1px solid #2D3748;border-radius:20px;padding:3px 10px;font-size:12px;color:#94A3B8;'>{row['category']}</span>
                            <span style='background:#212840;border:1px solid #2D3748;border-radius:20px;padding:3px 10px;font-size:12px;color:#94A3B8;'>{row['sentiment']}</span>
                            <span style='background:#212840;border:1px solid #2D3748;border-radius:20px;padding:3px 10px;font-size:12px;color:#94A3B8;'>📅 {row['date']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div style='background:#1A2030;border:1px solid #2D3748;border-radius:10px;padding:14px;margin-bottom:10px;text-align:center;'>
                        <div style='color:#6B7A99;font-size:10px;text-transform:uppercase;letter-spacing:1px;'>Crisis Score</div>
                        <div style='color:{"#EF4444" if score>=80 else "#F59E0B"};font-size:36px;font-weight:800;'>{score}</div>
                        <div style='color:#6B7A99;font-size:10px;'>/100</div>
                    </div>
                    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – ANALIZATOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔬 Analizator opinii pasażerów")
    st.markdown("Wklej dowolną opinię – model NLP oceni sentyment, kategorię i ryzyko kryzysu wizerunkowego.")

    example_texts = {
        "😡 Opóźnienie bez komunikacji": "Lot opóźniony 4 godziny, nikt nic nie mówi! Zero informacji od LOT. Siedzimy na lotnisku jak idioci. Skandal absolutny!",
        "🧳 Zaginiony bagaż": "Trzecia reklamacja za zaginiony bagaż i zero odpowiedzi od 3 tygodni. LOT to katastrofa obsługi klienta, nigdy więcej!",
        "💶 Odszkodowanie EC261": "Lot odwołany dzień przed wylotem, żadnej alternatywy. Żądam odszkodowania zgodnie z EC 261/2004. Prawnicy już zaangażowani.",
        "😊 Pozytywna opinia": "Fantastyczny lot do Nowego Jorku! Punktualnie, załoga bardzo miła i profesjonalna, jedzenie smaczne. Gorąco polecam LOT!",
        "😐 Neutralna opinia": "Lot był w porządku. Trochę się spóźnił ale nic strasznego. Obsługa OK, nic szczególnego.",
    }

    col_ex, _ = st.columns([3, 1])
    with col_ex:
        chosen = st.selectbox("Wczytaj przykład:", ["— wpisz własną opinię —"] + list(example_texts.keys()))

    user_text = example_texts.get(chosen, "") if chosen != "— wpisz własną opinię —" else ""
    opinion_input = st.text_area(
        "Treść opinii / tweeta / komentarza:",
        value=user_text,
        height=120,
        placeholder="Np. 'Lot opóźniony 3 godziny, obsługa ignorowała pasażerów, skandal!'"
    )

    analyze_btn = st.button("▶ Analizuj opinię", type="primary", use_container_width=False)

    if analyze_btn and opinion_input.strip():
        model = SentimentModel()
        scorer = CrisisScorer(threshold=crisis_threshold)

        with st.spinner("Analizuję..."):
            import time; time.sleep(0.4)
            result = model.predict_single(opinion_input)
            result = scorer.score_single(result)

        sentiment = result["sentiment"]
        crisis    = result["crisis_score"]
        category  = result["category"]
        conf      = result["confidence"]

        sent_color = {"negatywny": "#EF4444", "neutralny": "#F59E0B", "pozytywny": "#22C55E"}.get(sentiment, "#94A3B8")
        sent_emoji = {"negatywny": "😡", "neutralny": "😐", "pozytywny": "😊"}.get(sentiment, "🤔")
        crisis_color = "#EF4444" if crisis >= 70 else "#F59E0B" if crisis >= 40 else "#22C55E"
        urgency = "🔴 Wysoka" if crisis >= 70 else "🟡 Średnia" if crisis >= 40 else "🟢 Niska"

        st.divider()
        st.markdown("### Wyniki analizy NLP")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sentyment", f"{sent_emoji} {sentiment.upper()}")
        m2.metric("Crisis Score", f"{crisis}/100")
        m3.metric("Pilność", urgency)
        m4.metric("Pewność modelu", f"{conf}%")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(f"""
            <div style='background:#1A2030;border:1px solid #2D3748;border-radius:12px;padding:20px;'>
                <div style='color:#6B7A99;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Wynik sentymentu</div>
                <div style='font-size:48px;margin-bottom:4px;'>{sent_emoji}</div>
                <div style='color:{sent_color};font-size:28px;font-weight:800;letter-spacing:2px;'>{sentiment.upper()}</div>
                <div style='margin-top:16px;background:#212840;border-radius:8px;height:10px;overflow:hidden;'>
                    <div style='width:{conf}%;height:100%;background:{sent_color};border-radius:8px;'></div>
                </div>
                <div style='color:#6B7A99;font-size:12px;margin-top:6px;'>Pewność: {conf}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            recs = {
                "odszkodowanie": "Przekaż do działu odszkodowań. Poinformuj o prawach z EC 261/2004. Odpowiedź publiczna + prywatny kontakt w ciągu 2h.",
                "bagaż": "Poproś o numer PIR. Przekaż kontakt do działu bagażowego. Zapewnij priorytet sprawy i aktualizacje statusu.",
                "komunikacja": "Przeproś za brak komunikacji. Wyjaśnij przyczynę i opisz kolejne kroki. To kluczowy element dla pasażera.",
                "obsługa": "Przeproś i zapewnij o przekazaniu sprawy do supervisora. Poproś o szczegóły lotu i zaproponuj rekompensatę.",
                "opóźnienie": "Poinformuj o aktualnym statusie. Przypomnij o prawach pasażera i zaproponuj voucher na jedzenie/napoje.",
                "inne": "Odpowiedz publicznie z empatią. Zaproponuj kontakt prywatny celem rozwiązania problemu.",
            }
            rec_text = recs.get(category, recs["inne"]) if sentiment == "negatywny" else "Podziękuj za pozytywną opinię! Zaproś do ponownego skorzystania z LOT i zachęć do polecania znajomym."

            st.markdown(f"""
            <div style='background:#1A2030;border:1px solid #2D3748;border-radius:12px;padding:20px;height:100%;'>
                <div style='color:#6B7A99;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;'>Szczegóły analizy</div>
                <div style='display:flex;flex-direction:column;gap:10px;margin-bottom:16px;'>
                    <div style='background:#212840;border-radius:8px;padding:10px;display:flex;justify-content:space-between;'>
                        <span style='color:#6B7A99;font-size:13px;'>Kategoria problemu</span>
                        <span style='color:#E8EDF5;font-weight:700;font-size:13px;'>{category.upper()}</span>
                    </div>
                    <div style='background:#212840;border-radius:8px;padding:10px;display:flex;justify-content:space-between;'>
                        <span style='color:#6B7A99;font-size:13px;'>Crisis Score</span>
                        <span style='color:{crisis_color};font-weight:700;font-size:13px;'>{crisis}/100</span>
                    </div>
                    <div style='background:#212840;border-radius:8px;padding:10px;display:flex;justify-content:space-between;'>
                        <span style='color:#6B7A99;font-size:13px;'>Pilność reakcji</span>
                        <span style='font-weight:700;font-size:13px;'>{urgency}</span>
                    </div>
                </div>
                <div style='background:rgba(0,48,135,0.2);border:1px solid rgba(0,48,135,0.4);border-radius:8px;padding:14px;'>
                    <div style='color:#93C5FD;font-size:10px;font-family:monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;'>💡 Zalecana akcja PR</div>
                    <div style='color:#BFDBFE;font-size:13px;line-height:1.6;'>{rec_text}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("Wpisz treść opinii przed analizą.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – TABELA
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📋 Pełna tabela danych")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sent_filter = st.multiselect("Sentyment", ["negatywny", "neutralny", "pozytywny"],
                                     default=["negatywny", "neutralny", "pozytywny"])
    with col_f2:
        cat_filter = st.multiselect("Kategoria", df["category"].unique().tolist(),
                                    default=df["category"].unique().tolist())
    with col_f3:
        min_crisis = st.slider("Min. Crisis Score", 0, 100, 0)

    filtered = df[
        (df.sentiment.isin(sent_filter)) &
        (df.category.isin(cat_filter)) &
        (df.crisis_score >= min_crisis)
    ][["date", "text", "sentiment", "category", "crisis_score", "confidence"]].copy()

    filtered.columns = ["Data", "Treść opinii", "Sentyment", "Kategoria", "Crisis Score", "Pewność %"]
    filtered = filtered.sort_values("Crisis Score", ascending=False)

    st.markdown(f"**Wyniki: {len(filtered)} opinii**")
    st.dataframe(filtered, use_container_width=True, height=500)

    csv = filtered.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ Pobierz jako CSV",
        data=csv,
        file_name=f"lot_sentiment_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
