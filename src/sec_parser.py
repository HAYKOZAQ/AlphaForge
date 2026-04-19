import glob
import logging
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from src.utils.validators import validate_dataframe, EmbeddingSchema
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from concurrent.futures import ProcessPoolExecutor

def extract_text_from_10k(file_path: str, snippet_range: tuple) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    start, end = snippet_range
    return text[start:end]

def process_single_filing(fp, snippet_range):
    """Worker for individual filing extraction."""
    try:
        parts = Path(fp).parts
        ticker = parts[-4]
        extracted_text = extract_text_from_10k(fp, snippet_range)
        return {"ticker": ticker, "text_snippet": extracted_text}
    except Exception as e:
        logging.error(f"Failed to parse {fp}: {e}")
        return None

def main():
    config = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    sec_dir = repo_root / "data" / "sec_filings"
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    search_pattern = str(
        sec_dir / "sec-edgar-filings" / "*" / "10-K" / "*" / "full-submission.txt"
    )
    filing_paths = glob.glob(search_pattern)

    if not filing_paths:
        logging.warning("No SEC filings found. Run src/sec_ingest.py first.")
        return

    logging.info(f"Discovered {len(filing_paths)} 10-K filings. Parsing in parallel...")

    p_cfg = config['nlp']['sec_parser']
    snippet_range = (p_cfg['snippet_start'], p_cfg['snippet_end'])

    # Use ProcessPool for CPU-bound BeautifulSoup parsing
    from functools import partial
    worker_func = partial(process_single_filing, snippet_range=snippet_range)
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(worker_func, filing_paths))
    
    records = [r for r in results if r is not None]
    df = pd.DataFrame(records)

    logging.info(f"Loading sentence-transformer model: {p_cfg['model']}")
    model = SentenceTransformer(p_cfg['model'])

    logging.info("Generating semantic embeddings (batched)...")
    embeddings = model.encode(df["text_snippet"].tolist(), batch_size=32, show_progress_bar=True)

    embed_df = pd.DataFrame(
        embeddings,
        columns=[f"vec_{i}" for i in range(embeddings.shape[1])]
    )

    final_df = pd.concat([df[["ticker"]], embed_df], axis=1)
    final_df = final_df.groupby("ticker").mean().reset_index()

    n_clusters = min(p_cfg['n_clusters'], len(final_df))
    if n_clusters > 0:
        vector_cols = [c for c in final_df.columns if c.startswith("vec_")]
        km = KMeans(n_clusters=n_clusters, random_state=config['seed'], n_init=10)
        final_df["theme_cluster"] = km.fit_predict(final_df[vector_cols])
        logging.info("Assigned %s filing-embedding clusters across firms.", n_clusters)

    out_path = processed_dir / "sec_vector_embeddings.parquet"
    validate_dataframe(final_df, EmbeddingSchema, "SEC Vector Parsing")
    final_df.to_parquet(out_path, index=False)
    logging.info("Saved SEC vector embeddings to %s (shape=%s)", out_path, final_df.shape)

if __name__ == "__main__":
    main()