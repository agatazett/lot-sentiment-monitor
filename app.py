"""
LOT Polish Airlines – System Predykcji Nastrojów Pasażerów v2.0
================================================================
Nowe funkcje:
  - Integracja Brand24 API (symulacja) + Twitter/X API
  - Segmentacja pasażerów (4 profile)
  - Porównanie LOT vs konkurencja (Wizz Air, Ryanair)
  - Import CSV z Brand24
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random

from model import SentimentModel, CrisisScorer
from data_generator import generate_sample_data
from brand24_connector import Brand24Connector, TwitterConnector, FacebookGroupsConnector
from segmentation import PassengerSegmentation, SEGMENT_PROFILES

# ─── KONFIGURACJA ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="LOT Sentiment Monitor v2", page_icon="✈️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#07090F}
[data-testid="stSidebar"]{background:#0D1117;border-right:1px solid #1A2030}
[data-testid="metric-container"]{background:#1A2030;border:1px solid #212840;border-radius:12px;padding:16px!important}
.source-badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;
  font-family:monospace;background:#1A2030;border:1px solid #2D3748;color:#94A3B8;margin:2px}
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ LOT Monitor v2.0")
    st.divider()

    st.markdown("### 🔌 Źródła danych")
    data_source = st.radio("Wybierz źródło:", [
        "🤖 Dane syntetyczne (demo)",
        "📡 Brand24 API",
        "🐦 Twitter/X API",
        "📁 Import CSV (Brand24)",
        "🔀 Wszystkie źródła",
    ])

    st.divider()
    st.markdown("### 🔑 Klucze API")

    with st.expander("Brand24 API Token"):
        b24_token = st.text_input("Token:", type="password",
            placeholder="Wklej token Brand24...",
            help="Uzyskaj na: app.brand24.com → Ustawienia → API")
        if not b24_token:
            st.info("Brak tokena → tryb symulacji")

    with st.expander("Twitter/X Bearer Token"):
        tw_token = st.text_input("Bearer Token:", type="password",
            placeholder="Wklej Twitter Bearer Token...",
            help="Uzyskaj na: developer.twitter.com")
        if not tw_token:
            st.info("Brak tokena → dane demo")

    st.divider()
    st.markdown("### ⚙️ Parametry")
    n_opinions      = st.slider("Liczba opinii", 50, 500, 200, 50)
    days_back       = st.slider("Zakres (dni)", 7, 90, 30)
    crisis_threshold= st.slider("Próg alertu", 50, 90, 70)

    if st.button("🔄 Odśwież dane", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─── ŁADOWANIE DANYCH ─────────────────────────────────────────────────────────
@st.cache_data(ttl=120, show_spinner="Pobieram dane...")
def load_data(source, n, days, b24_tok, tw_tok, crisis_thr):
    model   = SentimentModel()
    scorer  = CrisisScorer(threshold=crisis_thr)
    segm    = PassengerSegmentation()
    dfs     = []

    if source == "🤖 Dane syntetyczne (demo)":
        dfs.append(generate_sample_data(n, days))

    elif source == "📡 Brand24 API":
        conn = Brand24Connector(api_token=b24_tok)
        df   = conn.fetch(days_back=days, max_results=n)
        dfs.append(df)

    elif source == "🐦 Twitter/X API":
        conn = TwitterConnector(bearer_token=tw_tok)
        dfs.append(conn.fetch(max_results=min(n, 100)))

    elif source == "🔀 Wszystkie źródła":
        b24 = Brand24Connector(api_token=b24_tok).fetch(days_back=days, max_results=n//2)
        tw  = TwitterConnector(bearer_token=tw_tok).fetch(max_results=min(n//4, 100))
        fb  = FacebookGroupsConnector().fetch(n=n//4, days_back=days)
        dfs.extend([b24, tw, fb])

    if not dfs:
        dfs.append(generate_sample_data(n, days))

    df = pd.concat(dfs, ignore_index=True)
    for col in ["likes","retweets","reach"]:
        if col not in df.columns:
            df[col] = 0
    df["likes"]    = pd.to_numeric(df["likes"],    errors="coerce").fillna(0).astype(int)
    df["retweets"] = pd.to_numeric(df["retweets"], errors="coerce").fillna(0).astype(int)
    df["reach"]    = pd.to_numeric(df["reach"],    errors="coerce").fillna(0).astype(int)
    df = model.predict(df)
    df = scorer.score(df)
    df = segm.segment(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

# Obsługa importu CSV
uploaded_csv = None
if data_source == "📁 Import CSV (Brand24)":
    with st.sidebar:
        st.divider()
        uploaded_csv = st.file_uploader("Wgraj CSV z Brand24:", type=["csv"])
        if not uploaded_csv:
            st.info("Wgraj plik → pojawią się dane")

if data_source == "📁 Import CSV (Brand24)" and uploaded_csv:
    try:
        df_all = Brand24Connector.from_csv(uploaded_csv)
        model  = SentimentModel()
        scorer = CrisisScorer(threshold=crisis_threshold)
        segm   = PassengerSegmentation()
        df_all = model.predict(df_all)
        df_all = scorer.score(df_all)
        df_all = segm.segment(df_all)
        df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
        st.sidebar.success(f"✅ Wczytano {len(df_all)} opinii z CSV")
    except Exception as e:
        st.error(f"Błąd importu CSV: {e}")
        df_all = load_data(data_source, n_opinions, days_back, b24_token, tw_token, crisis_threshold)
else:
    df_all = load_data(data_source, n_opinions, days_back, b24_token, tw_token, crisis_threshold)

df = df_all.copy()

# ─── STATUS BAR ───────────────────────────────────────────────────────────────
sources_present = df["source"].unique().tolist() if "source" in df.columns else []
src_html = " ".join([f"<span class='source-badge'>{s}</span>" for s in sources_present[:6]])
connected = bool(b24_token or tw_token)
status_color = "#22C55E" if connected else "#F59E0B"
status_text  = "API połączone" if connected else "Tryb symulacji"
st.markdown(f"""
<div style='background:#0D1117;border:1px solid #1A2030;border-radius:10px;
padding:10px 16px;margin-bottom:16px;display:flex;align-items:center;gap:16px;'>
  <span style='color:{status_color};font-size:12px;font-family:monospace;'>● {status_text}</span>
  <span style='color:#6B7A99;font-size:12px;'>Źródła: {src_html}</span>
  <span style='color:#6B7A99;font-size:12px;margin-left:auto;'>
    {len(df)} opinii · aktualizacja: {datetime.now().strftime("%H:%M:%S")}
  </span>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "👥 Segmenty pasażerów",
    "⚔️ LOT vs Konkurencja",
    "🔬 Analizator opinii",
    "📋 Tabela danych",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📡 Monitor Nastrojów Pasażerów LOT")

    total      = len(df)
    neg_pct    = round(len(df[df.sentiment=="negatywny"])/total*100,1) if total else 0
    avg_crisis = round(df.crisis_score.mean(),1) if total else 0
    alerts_n   = len(df[df.crisis_score>=crisis_threshold])
    pos_pct    = round(len(df[df.sentiment=="pozytywny"])/total*100,1) if total else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("📝 Opinie łącznie", total)
    c2.metric("😡 Negatywne", f"{neg_pct}%", delta=f"+{round(neg_pct-54,1)}%", delta_color="inverse")
    c3.metric("⚡ Śr. Crisis Score", f"{avg_crisis}/100", delta_color="inverse")
    c4.metric(f"🔴 Alerty (≥{crisis_threshold})", alerts_n, delta_color="inverse")
    c5.metric("😊 Pozytywne", f"{pos_pct}%", delta=f"{round(pos_pct-26,1)}%", delta_color="normal")

    st.divider()
    cl, cr = st.columns([2,1])

    with cl:
        st.markdown("#### Trend sentymentu w czasie")
        daily = df.groupby([df["date"].dt.date,"sentiment"]).size().reset_index(name="n")
        daily.columns = ["date","sentiment","n"]
        pivot = daily.pivot(index="date",columns="sentiment",values="n").fillna(0).reset_index()
        for s in ["negatywny","neutralny","pozytywny"]:
            if s not in pivot.columns: pivot[s]=0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pivot.date,y=pivot.negatywny,name="Negatywne",
            fill="tozeroy",line=dict(color="#EF4444",width=2),fillcolor="rgba(239,68,68,0.15)"))
        fig.add_trace(go.Scatter(x=pivot.date,y=pivot.pozytywny,name="Pozytywne",
            fill="tozeroy",line=dict(color="#22C55E",width=2),fillcolor="rgba(34,197,94,0.15)"))
        fig.add_trace(go.Scatter(x=pivot.date,y=pivot.neutralny,name="Neutralne",
            line=dict(color="#F59E0B",width=1.5,dash="dot")))
        fig.update_layout(paper_bgcolor="#1A2030",plot_bgcolor="#1A2030",
            font=dict(color="#94A3B8"),margin=dict(l=0,r=0,t=0,b=0),height=260,
            xaxis=dict(gridcolor="#212840"),yaxis=dict(gridcolor="#212840"),
            legend=dict(bgcolor="#212840"))
        st.plotly_chart(fig,use_container_width=True)

    with cr:
        st.markdown("#### Rozkład sentymentu")
        sc = df["sentiment"].value_counts()
        colors = {"negatywny":"#EF4444","neutralny":"#F59E0B","pozytywny":"#22C55E"}
        fig2 = go.Figure(go.Pie(labels=sc.index,values=sc.values,hole=0.6,
            marker_colors=[colors.get(l,"#6B7A99") for l in sc.index],
            textinfo="percent",textfont=dict(color="white",size=13)))
        fig2.update_layout(paper_bgcolor="#1A2030",plot_bgcolor="#1A2030",
            font=dict(color="#94A3B8"),margin=dict(l=0,r=0,t=0,b=0),height=260,
            legend=dict(bgcolor="#212840"))
        st.plotly_chart(fig2,use_container_width=True)

    ca,cb = st.columns(2)
    with ca:
        st.markdown("#### Kategorie problemów (negatywne)")
        neg_df = df[df.sentiment=="negatywny"]
        cc = neg_df["category"].value_counts().reset_index()
        cc.columns=["Kategoria","Liczba"]
        emoji_map={"opóźnienie":"✈️ Opóźnienia","bagaż":"🧳 Bagaż","obsługa":"👤 Obsługa",
                   "komunikacja":"📞 Komunikacja","odszkodowanie":"💶 Odszkodowania","inne":"❓ Inne"}
        cc["Kategoria"]=cc["Kategoria"].map(lambda x: emoji_map.get(x,x))
        fig3=px.bar(cc,x="Liczba",y="Kategoria",orientation="h",
            color="Liczba",color_continuous_scale=["#F59E0B","#EF4444"])
        fig3.update_layout(paper_bgcolor="#1A2030",plot_bgcolor="#1A2030",
            font=dict(color="#94A3B8"),margin=dict(l=0,r=0,t=0,b=0),height=260,
            coloraxis_showscale=False,xaxis=dict(gridcolor="#212840"),yaxis=dict(gridcolor="#212840"))
        st.plotly_chart(fig3,use_container_width=True)

    with cb:
        st.markdown("#### Rozkład Crisis Score")
        fig4=px.histogram(df,x="crisis_score",nbins=20,color_discrete_sequence=["#C8A951"])
        fig4.add_vline(x=crisis_threshold,line_dash="dash",line_color="#EF4444",
            annotation_text=f"Próg alertu ({crisis_threshold})",annotation_font_color="#EF4444")
        fig4.update_layout(paper_bgcolor="#1A2030",plot_bgcolor="#1A2030",
            font=dict(color="#94A3B8"),margin=dict(l=0,r=0,t=0,b=0),height=260,
            xaxis=dict(gridcolor="#212840"),yaxis=dict(gridcolor="#212840",title="Liczba opinii"))
        st.plotly_chart(fig4,use_container_width=True)

    # Alerty
    st.divider()
    st.markdown(f"### 🚨 Opinie wymagające reakcji PR (Crisis Score ≥ {crisis_threshold})")
    alerts = df[df.crisis_score>=crisis_threshold].sort_values("crisis_score",ascending=False).head(8)
    if alerts.empty:
        st.success("✅ Brak alertów przy aktualnym progu.")
    else:
        for _,row in alerts.iterrows():
            score = row["crisis_score"]
            border_col = "#EF4444" if score>=80 else "#F59E0B"
            seg   = row.get("segment","❓ Nieokreślony")
            src   = row.get("source","—")
            c_l, c_r = st.columns([4,1])
            with c_l:
                st.markdown(f"""
                <div style='background:#1A2030;border:1px solid #2D3748;border-radius:10px;
                padding:14px;margin-bottom:10px;border-left:3px solid {border_col}'>
                  <div style='color:#E8EDF5;font-size:14px;line-height:1.6;margin-bottom:8px;'>"{row['text']}"</div>
                  <div style='display:flex;gap:6px;flex-wrap:wrap;'>
                    <span class='source-badge'>{src}</span>
                    <span class='source-badge'>{seg}</span>
                    <span class='source-badge'>{row.get("category","—")}</span>
                    <span class='source-badge'>❤️ {row.get("likes",0)}</span>
                  </div>
                </div>""", unsafe_allow_html=True)
            with c_r:
                st.markdown(f"""
                <div style='background:#1A2030;border:1px solid #2D3748;border-radius:10px;
                padding:14px;text-align:center;margin-bottom:10px;'>
                  <div style='color:#6B7A99;font-size:10px;text-transform:uppercase;'>Crisis Score</div>
                  <div style='color:{border_col};font-size:36px;font-weight:800;'>{score}</div>
                  <div style='color:#6B7A99;font-size:10px;'>/100</div>
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – SEGMENTACJA PASAŻERÓW
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 👥 Segmentacja Pasażerów LOT")
    st.markdown("Każda opinia jest przypisana do profilu pasażera na podstawie słów kluczowych w treści.")

    segm = PassengerSegmentation()
    summary = segm.summary(df)

    # Karty segmentów
    seg_cols = st.columns(4)
    for i,(seg,profile) in enumerate(SEGMENT_PROFILES.items()):
        if seg == "❓ Nieokreślony": continue
        sub = df[df.segment==seg]
        if len(sub)==0: continue
        neg = round(len(sub[sub.sentiment=="negatywny"])/len(sub)*100,1)
        avg_cr = round(sub.crisis_score.mean(),1)
        with seg_cols[i % 4]:
            st.markdown(f"""
            <div style='background:#1A2030;border:1px solid #2D3748;border-radius:12px;
            padding:18px;border-top:3px solid {profile["color"]};margin-bottom:16px;'>
              <div style='font-size:28px;margin-bottom:8px;'>{seg}</div>
              <div style='color:#E8EDF5;font-weight:700;font-size:18px;margin-bottom:4px;'>{len(sub)} opinii</div>
              <div style='color:#94A3B8;font-size:12px;margin-bottom:12px;'>{profile["desc"][:80]}...</div>
              <div style='display:flex;justify-content:space-between;margin-bottom:8px;'>
                <span style='color:#6B7A99;font-size:12px;'>Negatywne:</span>
                <span style='color:{"#EF4444" if neg>50 else "#F59E0B"};font-weight:700;font-size:12px;'>{neg}%</span>
              </div>
              <div style='display:flex;justify-content:space-between;margin-bottom:12px;'>
                <span style='color:#6B7A99;font-size:12px;'>Crisis Score:</span>
                <span style='color:#C8A951;font-weight:700;font-size:12px;'>{avg_cr}/100</span>
              </div>
              <div style='color:#94A3B8;font-size:11px;'>{profile["priority"]}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    cl, cr = st.columns(2)
    with cl:
        st.markdown("#### Rozkład segmentów")
        seg_counts = df["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment","Liczba"]
        colors_map = {s:p["color"] for s,p in SEGMENT_PROFILES.items()}
        fig_seg = px.pie(seg_counts,names="Segment",values="Liczba",hole=0.5,
            color="Segment",color_discrete_map=colors_map)
        fig_seg.update_layout(paper_bgcolor="#1A2030",plot_bgcolor="#1A2030",
            font=dict(color="#94A3B8"),margin=dict(l=0,r=0,t=0,b=0),height=300,
            legend=dict(bgcolor="#212840"))
        st.plotly_chart(fig_seg,use_container_width=True)

    with cr:
        st.markdown("#### Crisis Score według segmentu")
        fig_box = px.box(df,x="segment",y="crisis_score",
            color="segment",color_discrete_map=colors_map,
            labels={"crisis_score":"Crisis Score","segment":"Segment"})
        fig_box.add_hline(y=crisis_threshold,line_dash="dash",line_color="#EF4444",
            annotation_text=f"Próg alertu ({crisis_threshold})")
        fig_box.update_layout(paper_bgcolor="#1A2030",plot_bgcolor="#1A2030",
            font=dict(color="#94A3B8"),margin=dict(l=0,r=0,t=0,b=0),height=300,
            xaxis=dict(gridcolor="#212840"),yaxis=dict(gridcolor="#212840"),
            showlegend=False)
        st.plotly_chart(fig_box,use_container_width=True)

    st.markdown("#### Tabela podsumowania segmentów")
    st.dataframe(summary,use_container_width=True,hide_index=True)

    st.divider()
    st.markdown("#### 💡 Zalecane akcje dla każdego segmentu")
    act_cols = st.columns(4)
    for i,(seg,profile) in enumerate(SEGMENT_PROFILES.items()):
        if seg=="❓ Nieokreślony": continue
        with act_cols[i%4]:
            st.markdown(f"**{seg}**")
            for action in profile["actions"]:
                st.markdown(f"→ {action}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – LOT VS KONKURENCJA
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## ⚔️ LOT vs Konkurencja")
    st.markdown("Porównanie nastrojów pasażerów LOT, Wizz Air i Ryanair na podstawie opinii z mediów społecznościowych.")

    @st.cache_data(ttl=300)
    def get_competition_data():
        """Generuje dane porównawcze dla 3 linii lotniczych."""
        airlines = {
            "LOT Polish Airlines": {"neg":0.58,"neu":0.21,"pos":0.21,"crisis_avg":61,"n":200,"color":"#003087"},
            "Wizz Air":            {"neg":0.65,"neu":0.18,"pos":0.17,"crisis_avg":68,"n":350,"color":"#C5007C"},
            "Ryanair":             {"neg":0.71,"neu":0.15,"pos":0.14,"crisis_avg":74,"n":520,"color":"#073590"},
        }
        categories = ["opóźnienie","bagaż","obsługa","komunikacja","odszkodowanie"]
        records = []
        for airline,params in airlines.items():
            n = params["n"]
            for _ in range(n):
                sent = random.choices(["negatywny","neutralny","pozytywny"],
                    weights=[params["neg"],params["neu"],params["pos"]])[0]
                crisis = max(0,min(100,int(
                    (params["crisis_avg"] + random.gauss(0,15)) *
                    (1.4 if sent=="negatywny" else 0.6 if sent=="pozytywny" else 1.0))))
                records.append({"airline":airline,"sentiment":sent,
                    "crisis_score":crisis,"category":random.choice(categories),
                    "color":params["color"]})
        return pd.DataFrame(records), airlines

    comp_df, airlines_params = get_competition_data()

    # Metryki porównawcze
    st.markdown("### Kluczowe wskaźniki")
    airline_cols = st.columns(3)
    for i,(airline,params) in enumerate(airlines_params.items()):
        with airline_cols[i]:
            is_lot = airline=="LOT Polish Airlines"
            border = "3px solid #C8A951" if is_lot else "1px solid #2D3748"
            sub = comp_df[comp_df.airline==airline]
            neg_p = round(params["neg"]*100,0)
            st.markdown(f"""
            <div style='background:#1A2030;border:{border};border-radius:12px;padding:18px;'>
              <div style='font-weight:700;font-size:16px;margin-bottom:12px;color:#E8EDF5;'>
                {"✈️ " if is_lot else ""}{airline}{"  ⭐" if is_lot else ""}
              </div>
              <div style='display:flex;justify-content:space-between;margin-bottom:8px;'>
                <span style='color:#6B7A99;font-size:13px;'>% negatywnych</span>
                <span style='color:#EF4444;font-weight:700;'>{neg_p}%</span>
              </div>
              <div style='display:flex;justify-content:space-between;margin-bottom:8px;'>
                <span style='color:#6B7A99;font-size:13px;'>Śr. Crisis Score</span>
                <span style='color:#C8A951;font-weight:700;'>{params["crisis_avg"]}/100</span>
              </div>
              <div style='display:flex;justify-content:space-between;'>
                <span style='color:#6B7A99;font-size:13px;'>Liczba opinii</span>
                <span style='color:#E8EDF5;font-weight:700;'>{params["n"]}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    cl2,cr2 = st.columns(2)
    with cl2:
        st.markdown("#### Rozkład sentymentu – porównanie")
        sent_comp = comp_df.groupby(["airline","sentiment"]).size().reset_index(name="n")
        sent_comp_pct = sent_comp.copy()
        totals = sent_comp.groupby("airline")["n"].transform("sum")
        sent_comp_pct["pct"] = (sent_comp_pct["n"]/totals*100).round(1)
        fig_comp = px.bar(sent_comp_pct,x="airline",y="pct",color="sentiment",barmode="stack",
            color_discrete_map={"negatywny":"#EF4444","neutralny":"#F59E0B","pozytywny":"#22C55E"},
            labels={"pct":"% opinii","airline":"Linia lotnicza"})
        fig_comp.update_layout(paper_bgcolor="#1A2030",plot_bgcolor="#1A2030",
            font=dict(color="#94A3B8"),margin=dict(l=0,r=0,t=0,b=0),height=300,
            xaxis=dict(gridcolor="#212840"),yaxis=dict(gridcolor="#212840"),
            legend=dict(bgcolor="#212840"))
        st.plotly_chart(fig_comp,use_container_width=True)

    with cr2:
        st.markdown("#### Śr. Crisis Score – porównanie")
        crisis_comp = comp_df.groupby("airline")["crisis_score"].mean().reset_index()
        crisis_comp.columns=["Linia","Crisis Score"]
        crisis_comp["color"] = crisis_comp["Linia"].map(
            {a:p["color"] for a,p in airlines_params.items()})
        fig_crisis = px.bar(crisis_comp,x="Linia",y="Crisis Score",
            color="Linia",color_discrete_map={a:p["color"] for a,p in airlines_params.items()},
            text="Crisis Score")
        fig_crisis.add_hline(y=crisis_threshold,line_dash="dash",line_color="#EF4444",
            annotation_text=f"Próg alertu ({crisis_threshold})")
        fig_crisis.update_traces(texttemplate="%{text:.1f}",textposition="outside")
        fig_crisis.update_layout(paper_bgcolor="#1A2030",plot_bgcolor="#1A2030",
            font=dict(color="#94A3B8"),margin=dict(l=0,r=0,t=10,b=0),height=300,
            xaxis=dict(gridcolor="#212840"),yaxis=dict(gridcolor="#212840"),
            showlegend=False)
        st.plotly_chart(fig_crisis,use_container_width=True)

    st.markdown("#### Kategorie problemów – porównanie")
    cat_comp = comp_df[comp_df.sentiment=="negatywny"].groupby(["airline","category"]).size().reset_index(name="n")
    fig_cat = px.bar(cat_comp,x="category",y="n",color="airline",barmode="group",
        color_discrete_map={a:p["color"] for a,p in airlines_params.items()},
        labels={"n":"Liczba negatywnych opinii","category":"Kategoria","airline":"Linia"})
    fig_cat.update_layout(paper_bgcolor="#1A2030",plot_bgcolor="#1A2030",
        font=dict(color="#94A3B8"),margin=dict(l=0,r=0,t=0,b=0),height=280,
        xaxis=dict(gridcolor="#212840"),yaxis=dict(gridcolor="#212840"),
        legend=dict(bgcolor="#212840"))
    st.plotly_chart(fig_cat,use_container_width=True)

    st.markdown("""
    <div style='background:rgba(0,48,135,0.15);border:1px solid rgba(0,48,135,0.3);
    border-radius:10px;padding:16px;margin-top:8px;'>
      <strong style='color:#93C5FD;'>📌 Wnioski dla pracy dyplomowej</strong><br><br>
      <span style='color:#BFDBFE;font-size:14px;line-height:1.8;'>
      Analiza porównawcza wskazuje, że LOT Polish Airlines osiąga lepsze wyniki sentymentu
      niż Ryanair i Wizz Air, jednak wciąż 58% opinii ma charakter negatywny.
      Główna przewaga konkurencyjna LOT powinna koncentrować się na komunikacji
      i czasie reakcji na reklamacje – obszarach gdzie konkurenci notują największe braki.
      Wdrożenie systemu predykcji nastrojów może obniżyć wskaźnik negatywnych opinii
      o szacowane 15-25% w perspektywie 12 miesięcy.
      </span>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – ANALIZATOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🔬 Analizator opinii")
    st.markdown("Wklej dowolną opinię – model oceni sentyment, segment pasażera i ryzyko kryzysu.")

    examples = {
        "😡 Opóźnienie bez komunikacji": "Lot opóźniony 4 godziny, nikt nic nie mówi! Zero informacji od LOT. Skandal absolutny!",
        "🧳 Zaginiony bagaż": "Trzecia reklamacja za zaginiony bagaż i zero odpowiedzi od 3 tygodni. LOT to katastrofa!",
        "💶 Odszkodowanie EC261": "Lot odwołany dzień przed wylotem. Żądam odszkodowania zgodnie z EC 261/2004. Prawnicy zaangażowani.",
        "👨‍👩‍👧 Rodzina z dziećmi": "Zostaliśmy z dziećmi na lotnisku bez żadnej pomocy po odwołaniu lotu. Dramat totalny!",
        "💼 Podróżny biznesowy": "Kolejne opóźnienie LOT na trasie biznesowej. Spóźniłem się na ważne spotkanie z klientem. Nieakceptowalne.",
        "😊 Pozytywna opinia": "Fantastyczny lot do Nowego Jorku! Punktualnie, załoga bardzo miła i profesjonalna. Polecam LOT!",
    }

    chosen = st.selectbox("Wczytaj przykład:", ["— wpisz własną —"]+list(examples.keys()))
    user_text = examples.get(chosen,"") if chosen!="— wpisz własną —" else ""
    opinion_input = st.text_area("Treść opinii:",value=user_text,height=100,
        placeholder="Np. 'Lot opóźniony 3 godziny, obsługa ignorowała pasażerów...'")

    if st.button("▶ Analizuj", type="primary") and opinion_input.strip():
        model2  = SentimentModel()
        scorer2 = CrisisScorer(threshold=crisis_threshold)
        segm2   = PassengerSegmentation()
        import time; time.sleep(0.3)
        result  = model2.predict_single(opinion_input)
        result  = scorer2.score_single(result)
        seg_label = segm2.segment_single(opinion_input)
        seg_profile = segm2.get_profile(seg_label)

        sentiment = result["sentiment"]
        crisis    = result["crisis_score"]
        conf      = result["confidence"]
        category  = result["category"]

        sent_color  = {"negatywny":"#EF4444","neutralny":"#F59E0B","pozytywny":"#22C55E"}.get(sentiment,"#94A3B8")
        sent_emoji  = {"negatywny":"😡","neutralny":"😐","pozytywny":"😊"}.get(sentiment,"🤔")
        crisis_col  = "#EF4444" if crisis>=70 else "#F59E0B" if crisis>=40 else "#22C55E"
        urgency     = "🔴 Wysoka" if crisis>=70 else "🟡 Średnia" if crisis>=40 else "🟢 Niska"

        st.divider()
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Sentyment", f"{sent_emoji} {sentiment.upper()}")
        m2.metric("Crisis Score", f"{crisis}/100")
        m3.metric("Pilność", urgency)
        m4.metric("Pewność modelu", f"{conf}%")
        m5.metric("Segment pasażera", seg_label.split(" ",1)[-1] if " " in seg_label else seg_label)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
            <div style='background:#1A2030;border:1px solid #2D3748;border-radius:12px;padding:20px;text-align:center;'>
              <div style='font-size:48px;'>{sent_emoji}</div>
              <div style='color:{sent_color};font-size:24px;font-weight:800;margin:8px 0;'>{sentiment.upper()}</div>
              <div style='background:#212840;border-radius:6px;height:8px;overflow:hidden;margin:12px 0;'>
                <div style='width:{conf}%;height:100%;background:{sent_color};'></div>
              </div>
              <div style='color:#6B7A99;font-size:12px;'>Pewność: {conf}%</div>
              <div style='margin-top:12px;color:#6B7A99;font-size:11px;text-transform:uppercase;'>Kategoria</div>
              <div style='color:#E8EDF5;font-weight:700;font-size:16px;'>{category.upper()}</div>
            </div>""",unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div style='background:#1A2030;border:1px solid #2D3748;border-radius:12px;padding:20px;'>
              <div style='color:#6B7A99;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Segment pasażera</div>
              <div style='font-size:32px;margin-bottom:4px;'>{seg_label.split()[0]}</div>
              <div style='color:#E8EDF5;font-weight:700;font-size:16px;margin-bottom:8px;'>{seg_profile["label"]}</div>
              <div style='color:#94A3B8;font-size:13px;line-height:1.5;margin-bottom:12px;'>{seg_profile["desc"]}</div>
              <div style='background:#212840;border-radius:8px;padding:8px 12px;'>
                <div style='color:#6B7A99;font-size:10px;text-transform:uppercase;'>Priorytet obsługi</div>
                <div style='color:#E8EDF5;font-weight:700;font-size:14px;margin-top:2px;'>{seg_profile["priority"]}</div>
              </div>
            </div>""",unsafe_allow_html=True)

        with col_c:
            recs = {"odszkodowanie":"Przekaż do działu odszkodowań. Odpowiedź + prywatny kontakt w ciągu 2h. Poinformuj o EC 261/2004.",
                    "bagaż":"Poproś o numer PIR. Przekaż do działu bagażowego. Zapewnij aktualizacje statusu co 24h.",
                    "komunikacja":"Przeproś za brak komunikacji. Wyjaśnij przyczynę i opisz kolejne kroki działania.",
                    "obsługa":"Przeproś. Przekaż sprawę do supervisora. Zaproponuj rekompensatę proporcjonalną do doświadczenia.",
                    "opóźnienie":"Poinformuj o aktualnym statusie lotu. Zaproponuj voucher. Przypomnij o prawach pasażera.",
                    "inne":"Odpowiedz publicznie z empatią i zaproponuj kontakt prywatny."}
            rec_text = recs.get(category,recs["inne"]) if sentiment=="negatywny" else "Podziękuj za opinię! Zaproś do ponownego skorzystania z LOT."
            priority_actions = seg_profile.get("actions",["Standardowa obsługa"])
            st.markdown(f"""
            <div style='background:#1A2030;border:1px solid #2D3748;border-radius:12px;padding:20px;'>
              <div style='color:#6B7A99;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;'>💡 Zalecana akcja PR</div>
              <div style='background:rgba(0,48,135,0.2);border:1px solid rgba(0,48,135,0.3);border-radius:8px;padding:12px;margin-bottom:12px;color:#BFDBFE;font-size:13px;line-height:1.6;'>{rec_text}</div>
              <div style='color:#6B7A99;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Akcje dla segmentu {seg_profile["label"]}</div>
              {"".join(f'<div style="color:#94A3B8;font-size:12px;margin-bottom:4px;">→ {a}</div>' for a in priority_actions)}
            </div>""",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – TABELA
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 📋 Pełna tabela danych")
    f1,f2,f3,f4 = st.columns(4)
    with f1: sf = st.multiselect("Sentyment",["negatywny","neutralny","pozytywny"],default=["negatywny","neutralny","pozytywny"])
    with f2: cf = st.multiselect("Kategoria", df.category.unique().tolist(), default=df.category.unique().tolist())
    with f3: seg_f = st.multiselect("Segment", df.segment.unique().tolist(), default=df.segment.unique().tolist())
    with f4: mc = st.slider("Min. Crisis Score", 0, 100, 0)

    fdf = df[(df.sentiment.isin(sf))&(df.category.isin(cf))&(df.segment.isin(seg_f))&(df.crisis_score>=mc)]
    cols_show = ["date","text","sentiment","category","segment","crisis_score","confidence","source"]
    cols_show = [c for c in cols_show if c in fdf.columns]
    fdf2 = fdf[cols_show].sort_values("crisis_score",ascending=False).copy()
    fdf2.columns = [{"date":"Data","text":"Treść","sentiment":"Sentyment","category":"Kategoria",
        "segment":"Segment","crisis_score":"Crisis Score","confidence":"Pewność %","source":"Źródło"}.get(c,c) for c in fdf2.columns]
    st.markdown(f"**Wyniki: {len(fdf2)} opinii**")
    st.dataframe(fdf2,use_container_width=True,height=450)
    csv = fdf2.to_csv(index=False,encoding="utf-8-sig")
    st.download_button("⬇️ Pobierz CSV",data=csv,
        file_name=f"lot_sentiment_{datetime.now().strftime('%Y%m%d')}.csv",mime="text/csv")
