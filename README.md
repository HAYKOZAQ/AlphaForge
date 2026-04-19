# AI-Augmented Quantitative Market Development Intelligence System 🚀

This repository contains a **PhD-Level, Institutional-Grade Research Platform** for AI-augmented market development analysis. It models **sector-level market development acceleration** and **firm-level strategic interconnectedness** by fusing quantitative signals with alternative data: SEC filing sentiment, options skew, labor market demand, and insider trading conviction.

The pipeline identifies the policy, macro, and human drivers that influence future market participation, liquidity, and relative strength. 

This platform has been rigorously engineered for **production deployment**, featuring robust Directed Acyclic Graph (DAG) orchestration, a highly-available REST API microservice, completely vectorized graph computation, and automated CI/CD.

## ✨ Project Architecture & Core Methodologies

The system combines **seven complementary domains** to generate high-conviction market intelligence:

1. **Quantitative Data Engineering**: OHLCV data for sector ETFs and S&P 500 constituents.
2. **Macro Regime Switching**: Fitting a **Hidden Markov Model (HMM)** on macroeconomic indicators (Fed Funds, CPI, WTI, Unemployment) to detect dynamic market regimes and dynamically adjust inference logic.
3. **Alternative Alpha Dimensions**:
    - **Insider Trading (Form 4)**: Capturing "Open Market" buy/sell conviction from corporate executives.      
    - **Labor Market (JOLTS)**: Sector-level hiring momentum vs. the broader economy.
    - **Options Skew**: Measuring tail-risk hedging via Put/Call open interest and volume z-scores.
3. **NLP & Semantic Processing**: Extracting sentiment and thematic clusters from Earnings Call Transcripts and 10-K filings using `sentence-transformers`.
4. **Strategic Multi-Layer Network Graph (Vectorized)**: 
    - **Semantic Layer**: 10-K embedding similarity.
    - **Capital Layer**: Common institutional ownership (13F overlap computed via $O(1)$ vectorized matrix intersection).
    - **Human Layer**: Shared insiders and strategic cross-mentions.
5. **Graph Network Analytics**: Computing **PageRank** and **Eigenvector Centrality** to identify strategic hubs and "hidden" market connections.
6. **Combinatorial Purged Cross-Validation (CPCV)**: Training **Model C** (Full Alt-Data Pack) using Purged K-Fold Cross-Validation to eliminate serial correlation leakage, and evaluating via the **Deflated Sharpe Ratio (DSR)** to mathematically prove the strategy is not overfit.
7. **Causal Inference**: Utilizing Microsoft's `dowhy` library to estimate the Average Treatment Effect (ATE) via backdoor linear regression and placebo refutation, proving that signals like Insider Conviction structurally drive returns rather than just correlating.

## 🛠️ Production Tech Stack & MLOps

- **Data Ingestion**: `yfinance`, `edgartools`, `sec-edgar-downloader`, FRED API. Built with `tenacity` for exponential backoff and network resiliency.
- **Machine Learning**: `xgboost`, `scikit-learn`, `MLflow` (Experiment Tracking & Model Registry).
- **DAG Orchestration**: **Prefect** manages the parallel execution and caching of ingestion, feature engineering, and training tasks.
- **Serving Layer**: A **FastAPI** microservice automatically caches the latest MLflow models to expose a `<host>:8000/predict/sector` endpoint.
- **CI/CD & Reproducibility**: `pip-tools` (`requirements.in`) guarantees mathematically locked environments. GitHub Actions automates `pytest` and `ruff` linting, followed by a Docker image build.
- **Infrastructure**: `Streamlit`, `Plotly`, `Docker`, `Docker Compose`.

---

## ⚙️ Pipeline Execution Phases

The repository is modular and orchestrated via Prefect in `src/prefect_flow.py`:

### Phase I: Multi-Source Parallel Ingestion
- `src/data_ingest.py`: OHLCV and Options data.
- `src/macro_loader.py`: FRED macroeconomic and JOLTS labor indicators.
- `src/insider_ingest.py`: SEC Form 4 insider transactions.
- `src/transcript_ingest.py`: Earnings call transcript collection.

### Phase II: Feature Engineering & Graph Analytics
- `src/feature_builder.py`: Constructs 45+ features across Quant, Macro, NLP, and Human domains.
- `src/graph_builder.py`: Constructs the completely vectorized strategic multi-layer network.
- `src/target_builder.py`: Aligns data into a unified, monthly-resampled research dataset.

### Phase III: Modeling, Validation, & Serving
- `src/train.py`: Rolling Walk-Forward sector model training with Shapley explanations.
- `src/backtest.py`: Final institutional validation and metrics export.
- `src/api/main.py`: The live FastAPI inference server.
- `app/streamlit_app.py`: Interactive dashboard and strategic network visualization.

---

## 📊 Performance Benchmarks
*Current validation results on the focus tech sector universe (Expanding Window CV):*

| Model | Alpha Pack | AUC | Sharpe Ratio |
| :--- | :--- | :--- | :--- |
| **Model A** | Quant Only | 0.76 | -0.44 |
| **Model B** | +Macro & Labor | 0.91 | 0.11 |
| **Model C** | **Full Alt-Data Pack** | **0.91** | **0.20** |

---

## 🚀 How to Run (Docker Deployment)

The entire platform is containerized. 

1. **Clone the repository and set up environment variables**:
   ```bash
   cp .env.example .env
   # Add your FRED API Key to the .env file
   ```
2. **Launch the platform via Docker Compose**:
   ```bash
   docker-compose up --build
   ```
   *This automatically starts the Training Pipeline, the Streamlit Dashboard (Port 8501), and the FastAPI Inference Server (Port 8000).*

3. **Access Services**:
   - **Research Dashboard**: `http://localhost:8501`
   - **FastAPI Documentation (Swagger)**: `http://localhost:8000/docs`

## 💻 Local Development Setup

If you prefer to run the orchestration locally instead of inside Docker:

```bash
# Install exact dependencies
pip install pip-tools
pip-sync requirements.txt

# Run the Prefect DAG pipeline
python src/prefect_flow.py

# Launch FastAPI Server
uvicorn src.api.main:app --reload

# Run Tests
pytest tests/
```

---

## 🚦 Disclosures
- **Survivorship Bias**: Currently uses current S&P 500 constituents.
- **SEC Compliance**: Identity for `edgartools` and `sec-edgar-downloader` is managed via `src/utils/config_loader.py`.
