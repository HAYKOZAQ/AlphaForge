import json
import pickle
import warnings
from pathlib import Path

import google.generativeai as genai
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Market Development Research Console", layout="wide")

# Institutional Aesthetic Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top left, #0e1117, #161b22);
    }
    
    /* Card Styling */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 212, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        border-color: #00d4ff;
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Global Header Gradients */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00d4ff, #0055ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .stDivider {
        border-bottom: 2px solid rgba(0, 212, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

repo_root = Path(__file__).resolve().parent.parent
processed_dir = repo_root / "data" / "processed"
models_dir = repo_root / "models"
graph_dir = repo_root / "data" / "graph_models"
reports_dir = repo_root / "reports"


def get_mtime(path):
    return path.stat().st_mtime if path.exists() else 0

@st.cache_data
def load_sector_data(mtime):
    df_path = processed_dir / "master_dataset.parquet"
    if df_path.exists():
        return pd.read_parquet(df_path)
    return pd.DataFrame()


@st.cache_resource
def load_sector_model(mtime):
    model_path = models_dir / "model_C.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_graph(mtime):
    graph_path = graph_dir / "strategic_network.gml"
    if graph_path.exists():
        return nx.read_gml(str(graph_path))
    return None


@st.cache_data
def load_firm_data(mtime):
    df_path = processed_dir / "master_firm_dataset.parquet"
    if df_path.exists():
        return pd.read_parquet(df_path)
    return pd.DataFrame()


@st.cache_resource
def load_firm_model(mtime):
    model_path = models_dir / "firm_xgboost_core.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_sector_metrics(mtime):
    path = reports_dir / "metrics" / "walk_forward_metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_firm_metrics(mtime):
    path = reports_dir / "firm_metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_shap_data(mtime):
    shap_path = reports_dir / "shap_values.parquet"
    base_path = reports_dir / "shap_base_features.parquet"
    if shap_path.exists() and base_path.exists():
        return pd.read_parquet(shap_path), pd.read_parquet(base_path)
    return None, None


def render_shap_beeswarm_plotly(shap_df, base_df):
    """
    Implements a custom Plotly Beeswarm plot mimicking SHAP's summary_plot.
    """
    import numpy as np
    
    # Global feature importance (mean absolute SHAP)
    importance = shap_df.abs().mean().sort_values(ascending=True)
    features = importance.index.tolist()
    
    plot_data = []
    
    for i, feature in enumerate(features):
        shaps = shap_df[feature].values
        features_raw = base_df[feature].values
        
        # Normalize features for colormap (0 to 1)
        f_min, f_max = features_raw.min(), features_raw.max()
        if f_max > f_min:
            colors = (features_raw - f_min) / (f_max - f_min)
        else:
            colors = np.zeros_like(features_raw)
            
        # Add jitter to Y-axis for beeswarm effect
        jitter = np.random.normal(0, 0.12, size=len(shaps))
        
        df_temp = pd.DataFrame({
            'shap': shaps,
            'feature_val': features_raw,
            'norm_color': colors,
            'jitter_y': i + jitter,
            'feature_name': feature
        })
        plot_data.append(df_temp)
        
    df_plot = pd.concat(plot_data)
    
    fig = px.scatter(
        df_plot,
        x='shap',
        y='jitter_y',
        color='norm_color',
        color_continuous_scale='RdYlBu_r', # Red for high, Blue for low
        hover_data={'shap': ':.4f', 'feature_val': ':.4f', 'feature_name': True, 'jitter_y': False, 'norm_color': False},
        labels={'shap': 'SHAP Value (Impact on Model Output)', 'norm_color': 'Feature Value'}
    )
    
    fig.update_layout(
        height=len(features) * 40 + 100,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(features))),
            ticktext=features,
            title='Features (Sorted by Importance)',
            gridcolor='rgba(255,255,255,0.05)'
        ),
        xaxis=dict(
            title='SHAP Value (Impact on Prediction)',
            zeroline=True,
            zerolinecolor='white',
            gridcolor='rgba(255,255,255,0.05)'
        ),
        coloraxis_colorbar=dict(
            title="Feature Value",
            tickvals=[0, 1],
            ticktext=["Low", "High"]
        ),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    # Optimize marker size
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='rgba(0,0,0,0.2)')))
    
    return fig


def render_metrics_block(metrics: dict, title: str):
    st.markdown(f"## {title}")
    if not metrics:
        st.info("No saved walk-forward metrics file found. Run 'make all' to generate.")
        return

    for model_name, vals in metrics.items():
        auc_val = vals.get("auc")
        f1_val = vals.get("f1")
        rec_val = vals.get("recall")
        acc_val = vals.get("accuracy")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{model_name} AUC", f"{auc_val:.4f}" if isinstance(auc_val, (int, float)) else "N/A")
        c2.metric(f"{model_name} Accuracy", f"{acc_val:.4f}" if isinstance(acc_val, (int, float)) else "N/A")
        c3.metric(f"{model_name} F1 Score", f"{f1_val:.4f}" if isinstance(f1_val, (int, float)) else "N/A")
        c4.metric(f"{model_name} Recall", f"{rec_val:.4f}" if isinstance(rec_val, (int, float)) else "N/A")
        
        with st.expander(f"Detailed Matrix for {model_name}"):
            cm = vals.get("conf_matrix")
            if cm:
                st.write("Confusion Matrix:")
                st.write(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]))


def main():
    with st.sidebar:
        st.header("Research Synchronization")
        
        # Monitor critical path mtimes
        p_mtime = get_mtime(processed_dir / "master_dataset.parquet")
        g_mtime = get_mtime(graph_dir / "strategic_network.gml")
        m_mtime = get_mtime(reports_dir / "metrics" / "walk_forward_metrics.json")
        f_mtime = get_mtime(processed_dir / "master_firm_dataset.parquet")
        
        st.write(f"**Last Data Reset**: {pd.to_datetime(p_mtime, unit='s').strftime('%Y-%m-%d %H:%M')}")
        
        if st.button("Sync Research Data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Research cache purged. Visuals forced to latest artifacts.")
            st.rerun()

        st.divider()
        st.info(
            "Visuals automatically sync with artifacts upon page refresh. "
            "Use the button above only to manually force a deep cache purge."
        )

        gemini_api_key = st.text_input("Gemini API Key (Optional)", type="password")
        gemini_model = "gemini-1.5-flash"

        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            try:
                available_models = [
                    m.name
                    for m in genai.list_models()
                    if "generateContent" in m.supported_generation_methods
                ]
                if available_models:
                    default_idx = 0
                    gemini_model = st.selectbox("Select Gemini Model", available_models, index=default_idx)
                else:
                    st.warning("No compatible Gemini models found for this API key.")
            except Exception:
                st.error("API key invalid or Gemini service unavailable.")

    st.title("Market Development Research Console")
    st.markdown(
        """
        ---
        **Operational Status**: V2.0 Institutional Strategy Implemented  
        Advanced research engine fusing Sector Momentum, Macro Indicators, and SEC Semantic Signals.
        """
    )

    sector_metrics = load_sector_metrics(m_mtime)
    render_metrics_block(sector_metrics, "Sector Model Evaluation Summary")

    tab1, tab2, tab3 = st.tabs(["Sector Predictor", "SEC Similarity Graph", "Firm Predictor"])

    with tab1:
        df = load_sector_data(p_mtime)
        model_dict = load_sector_model(p_mtime)

        if df.empty or model_dict is None:
            st.warning("Sector data or sector model is missing. Run the pipeline first.")
        else:
            clf = model_dict["model"]
            features = model_dict["features"]

            df["date"] = pd.to_datetime(df["date"])
            latest_date = df["date"].max()
            df_latest = df[df["date"] == latest_date].copy()

            available_features = [f for f in features if f in df_latest.columns]

            if available_features:
                X_latest = df_latest[available_features]
                preds = clf.predict_proba(X_latest)[:, 1]
                df_latest["prediction_score"] = preds * 100.0

                ranked_cols = ["ticker", "prediction_score"] + available_features
                df_ranked = df_latest[ranked_cols].sort_values(by="prediction_score", ascending=False)

                st.header(f"Sector rankings as of {latest_date.strftime('%Y-%m')}")
                c1, c2 = st.columns([1, 2])

                with c1:
                    st.subheader("Current ranking")
                    display_df = df_ranked[["ticker", "prediction_score"]].copy()
                    display_df["prediction_score"] = display_df["prediction_score"].round(2)
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                with c2:
                    st.subheader("Score distribution")
                    fig = px.bar(
                        df_ranked,
                        x="ticker",
                        y="prediction_score",
                        color="prediction_score",
                        title="Current sector model score",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.header("Explainability & Attribution")
                c_expl1, c_expl2 = st.columns([3, 1])
                
                with c_expl1:
                    st.subheader("Global SHAP Attribution (Interactive)")
                    shap_df, base_df = load_shap_data(m_mtime)
                    if shap_df is not None:
                        fig_shap = render_shap_beeswarm_plotly(shap_df, base_df)
                        st.plotly_chart(fig_shap, use_container_width=True)
                    else:
                        shap_path = reports_dir / "plots" / "shap_summary.png"
                        if shap_path.exists():
                            st.image(str(shap_path), caption="Fallback: Static SHAP Plot")
                        else:
                            st.info("SHAP data not yet generated. Run the training pipeline.")
                
                with c_expl2:
                    st.subheader("Interactive Insights")
                    st.write("Each point represents a prediction instance across the latest deployment month.")
                    st.write("**X-Axis**: How much the feature moved the final score.")
                    st.write("**Color**: Relative intensity (Red = High, Blue = Low).")
                    st.markdown("""
                    > [!TIP]
                    > Hover over points to see the exact numeric feature value behind the signal.
                    """)

                st.divider()
                st.header("Sector feature inspection")

                selected_ticker = st.selectbox("Select a sector", df_ranked["ticker"].tolist())
                sector_row = df_ranked[df_ranked["ticker"] == selected_ticker].iloc[0]

                categories = [
                    "vol_20d",
                    "mom_3m",
                    "volume_zscore",
                    "ai_adoption",
                    "policy_support",
                    "capex_intent",
                    "sentiment",
                    "opt_put_call_oi_ratio_z",
                    "transcript_sentiment"
                ]
                valid_cats = [c for c in categories if c in sector_row.index]
                normalized_vals_dict = {}

                if valid_cats:
                    vals = sector_row[valid_cats].astype(float)
                    overall_min = df_latest[valid_cats].min()
                    overall_max = df_latest[valid_cats].max()
                    normalized_vals = (vals - overall_min) / (overall_max - overall_min + 1e-6)
                    normalized_vals_dict = {
                        cat: float(val) for cat, val in zip(valid_cats, normalized_vals)
                    }

                    fig2 = go.Figure()
                    fig2.add_trace(
                        go.Scatterpolar(
                            r=normalized_vals,
                            theta=valid_cats,
                            fill="toself",
                            name=selected_ticker,
                        )
                    )
                    fig2.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False,
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                st.divider()
                st.header("Advanced Alpha Signals")
                c_alpha1, c_alpha2 = st.columns(2)
                
                with c_alpha1:
                    st.subheader("Options Market Skew (Fear vs Greed)")
                    opt_oi_z = sector_row.get('opt_put_call_oi_ratio_z', 0)
                    
                    # Mapping Z-score to qualitative sentiment
                    if opt_oi_z > 1.5:
                        sentiment_label = "🔥 High Fear (Hedging Spike)"
                        color = "red"
                    elif opt_oi_z < -1.5:
                        sentiment_label = "🚀 High Greed (Call Dominance)"
                        color = "green"
                    else:
                        sentiment_label = "⚖️ Neutral"
                        color = "gray"
                        
                    st.markdown(f"**Current Signal**: <span style='color:{color}; font-size:1.2em;'>{sentiment_label}</span>", unsafe_allow_html=True)
                    st.write(f"The Put/Call OI Z-score is **{opt_oi_z:.2f}**. This indicates institutional positioning relative to historical norms.")

                with c_alpha2:
                    st.subheader("Quarterly Earnings Sentiment")
                    trans_sent = sector_row.get('transcript_sentiment', 0)
                    st.metric("Management Tone Score", f"{trans_sent:.2f}", delta=None)
                    st.progress((trans_sent + 1) / 2) # Map -1:1 to 0:1
                    st.write("Aggregated from focus ticker transcripts. Scores > 0 indicate optimistic forward-looking guidance.")

                st.divider()
                st.header("Labor Market Demand (JOLTS)")
                
                # Smart Fallback: Find the latest date with valid labor data
                if "labor_hiring_momentum" in df.columns:
                    valid_labor_df = df[(df["labor_hiring_momentum"] != 0) & (df["labor_hiring_momentum"].notna())]
                    if not valid_labor_df.empty:
                        target_labor_date = valid_labor_df["date"].max()
                        df_labor_latest = valid_labor_df[valid_labor_df["date"] == target_labor_date][["ticker", "labor_hiring_momentum"]].sort_values("labor_hiring_momentum", ascending=False)
                        
                        st.subheader(f"Relative Hiring Momentum (Latest Signal: {target_labor_date.strftime('%b %Y')})")
                        
                        fig_labor = px.bar(
                            df_labor_latest,
                            x="ticker",
                            y="labor_hiring_momentum",
                            color="labor_hiring_momentum",
                            color_continuous_scale="RdYlGn",
                            title=f"Hiring vs Economy Trend (as of {target_labor_date.strftime('%Y-%m')})",
                            labels={"labor_hiring_momentum": "Relative Momentum (3m)"}
                        )
                        st.plotly_chart(fig_labor, use_container_width=True)
                        st.write("Positive scores indicate sectors hiring faster than the broader economy. This signal typically lags price action by 30-60 days.")
                    else:
                        st.info("Labor data available in dataset but currently shows zero-intensity across all sectors.")
                else:
                    st.warning("Labor hiring momentum data is currently missing from the master dataset.")

                # 4. Insider Conviction
                st.divider()
                st.header("Institutional Confidence (Insider Activity)")
                if "insider_net_intensity" in df.columns:
                    valid_insider_df = df[(df["insider_net_intensity"] != 0) & (df["insider_net_intensity"].notna())]
                    if not valid_insider_df.empty:
                        target_insider_date = valid_insider_df["date"].max()
                        df_insider_latest = valid_insider_df[valid_insider_df["date"] == target_insider_date][["ticker", "insider_net_intensity"]].sort_values("insider_net_intensity", ascending=False)
                        
                        st.subheader(f"Net Insider Intensity (Latest Signal: {target_insider_date.strftime('%b %Y')})")
                        
                        fig_insider = px.bar(
                            df_insider_latest,
                            x="ticker",
                            y="insider_net_intensity",
                            color="insider_net_intensity",
                            color_continuous_scale="RdYlGn",
                            title=f"Insider Buy/Sell Conviction (as of {target_insider_date.strftime('%Y-%m')})",
                            labels={"insider_net_intensity": "Net Conviction Score"}
                        )
                        st.plotly_chart(fig_insider, use_container_width=True)
                        st.write("Scores closer to +1 indicate pure insider buying. We surface the latest cross-sectional signal regardless of reporting lag.")
                    else:
                        st.info("Insider conviction data available but empty for the latest period.")
                else:
                    st.warning("Insider conviction data missing from the master dataset.")

                st.subheader("Optional natural-language explanation")
                st.caption(
                    "Generated summaries are optional research aids and should not be treated as causal proof."
                )

                if gemini_api_key:
                    if st.button(f"Generate summary for {selected_ticker}"):
                        with st.spinner("Generating summary..."):
                            try:
                                model = genai.GenerativeModel(gemini_model)
                                prompt = f"""
                                You are a quantitative research assistant.

                                The sector ETF {selected_ticker} has a model score of
                                {sector_row.get('prediction_score', 0):.2f} out of 100.

                                Relative feature strengths:
                                {normalized_vals_dict}

                                Write a concise 3-4 sentence explanation of why this sector may
                                currently rank where it does, based only on the relative strength
                                of the available momentum, volatility, and text-derived features.
                                Avoid overstating causality.
                                """
                                response = model.generate_content(
                                    prompt,
                                    generation_config={"temperature": 0.1}
                                )
                                st.info(response.text)
                            except Exception as exc:
                                st.error(f"Gemini error: {exc}")
                else:
                    st.caption("Enter a Gemini API key in the sidebar to enable optional summaries.")

    with tab2:
        st.header("Strategic Multi-Layer Network Graph")
        st.markdown(
            "This unified graph combines **Semantic Similarity**, **Capital Overlap**, "
            "and **Human Links**. Nodes represent firms; edges represent strategic interconnectedness."
        )

        G = load_graph(g_mtime)
        if G:
            # Scale-Up Optimization: Filter for Top Hubs
            st.subheader("Filter Strategic Connections")
            cent_df = pd.read_parquet(processed_dir / "graph_centrality.parquet")
            max_hubs = len(G.nodes())
            num_hubs = st.slider("Strategic Focus: Show Top Hubs (PageRank)", 5, max_hubs, min(20, max_hubs))
            
            top_hubs = cent_df.sort_values("network_pagerank", ascending=False).head(num_hubs)["ticker"].tolist()
            G_sub = G.subgraph(top_hubs)

            net = Network(height="650px", width="100%", bgcolor="#111111", font_color="white")
            
            # Add nodes with sentiment-based colors
            for node in G_sub.nodes():
                net.add_node(node, label=node, title=f"Firm: {node}", color="#2196F3")

            # Add multi-layer edges
            for u, v, data in G_sub.edges(data=True):
                weight = data.get('weight', 1.0)
                # Determine edge color based on dominant layer
                edge_color = "#444444" 
                title = f"Total Weight: {weight:.2f}"
                
                if data.get('human', 0) > 0:
                    edge_color = "#00E676" 
                    title += "\nLayer: [Human Network] - Shared Insiders"
                elif data.get('capital', 0) > data.get('semantic', 0):
                    edge_color = "#FFD700" 
                    title += f"\nLayer: [Capital Network] - Institutional Overlap ({data.get('capital',0):.2f})"
                else:
                    edge_color = "#2979FF" 
                    title += f"\nLayer: [Semantic Network] - Filing Similarity ({data.get('semantic',0):.2f})"

                net.add_edge(u, v, value=weight, color=edge_color, title=title)

            net.repulsion(node_distance=250, spring_length=350)

            path_html = processed_dir / "graph_render.html"
            net.write_html(str(path_html))

            with open(path_html, "r", encoding="utf-8") as f:
                html_code = f.read()
            components.html(html_code, height=700)

            st.subheader("Network Strategic Hubs (PageRank)")
            cent_df = pd.read_parquet(processed_dir / "graph_centrality.parquet")
            if not cent_df.empty:
                st.dataframe(cent_df.sort_values("network_pagerank", ascending=False), 
                             use_container_width=True, hide_index=True)
            
            st.caption("Legend: 🔵 Semantic (SEC) | 🟡 Capital (Institutions) | 🟢 Human (Insiders)")
        else:
            st.warning("No graph file found. Run the Multi-Layer Graph stage first.")

    with tab3:
        st.header("Firm-level proof-of-concept predictor")
        st.markdown(
            "Firm outputs combine price-derived features, filing-derived graph centrality, "
            "and firm-level engineered signals."
        )

        firm_metrics = load_firm_metrics(f_mtime)
        if firm_metrics:
            st.subheader("Firm Model Evaluation Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Firm AUC", f"{firm_metrics.get('auc', 0):.4f}")
            c2.metric("Firm Accuracy", f"{firm_metrics.get('accuracy', 0):.4f}")
            c3.metric("Baseline Accuracy", f"{firm_metrics.get('baseline_accuracy', 0):.4f}")

        firm_df = load_firm_data(f_mtime)
        firm_dict = load_firm_model(f_mtime)

        if firm_df.empty or firm_dict is None:
            st.warning("Firm data or firm model is missing. Run the firm-level pipeline first.")
        else:
            clf_f = firm_dict["model"]
            features_f = firm_dict["features"]

            firm_df["date"] = pd.to_datetime(firm_df["date"])
            latest_f_date = firm_df["date"].max()
            df_f_latest = firm_df[firm_df["date"] == latest_f_date].copy()

            avail_f = [f for f in features_f if f in df_f_latest.columns]

            if avail_f:
                X_f = df_f_latest[avail_f]
                preds_f = clf_f.predict_proba(X_f)[:, 1]
                df_f_latest["firm_acceleration_score"] = preds_f * 100.0

                display_cols = ["ticker", "firm_acceleration_score"]
                if "network_centrality" in df_f_latest.columns:
                    display_cols.append("network_centrality")
                display_cols += [c for c in avail_f if "theme" in c]

                df_f_ranked = df_f_latest[display_cols].sort_values(
                    by="firm_acceleration_score",
                    ascending=False
                )

                st.subheader(f"Top 20 firms ({latest_f_date.strftime('%Y-%m-%d')})")
                st.dataframe(df_f_ranked.head(20), use_container_width=True, hide_index=True)

                if "network_centrality" in df_f_ranked.columns:
                    fig3 = px.scatter(
                        df_f_ranked.head(50),
                        x="network_centrality",
                        y="firm_acceleration_score",
                        size="firm_acceleration_score",
                        hover_data=["ticker"],
                        title="Network centrality vs firm model score",
                    )
                    st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    main()