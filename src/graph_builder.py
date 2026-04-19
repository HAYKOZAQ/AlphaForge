import logging
from pathlib import Path
import networkx as nx
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.config_loader import load_config
from src.utils.validators import validate_dataframe, CentralitySchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import numpy as np

def build_multi_layer_graph(df_embed, df_holders, df_insider, threshold=0.6, alpha=0.5, beta=0.4, gamma=0.1):
    logging.info("Building multi-layer strategic graph (Vectorized)...")
    
    # 1. Semantic Layer (Cosine Similarity of SEC Embeddings)
    vector_cols = [c for c in df_embed.columns if c.startswith("vec_")]
    tickers = df_embed["ticker"].tolist()
    vectors = df_embed[vector_cols].values
    sem_matrix = cosine_similarity(vectors)
    
    G = nx.Graph()
    G.add_nodes_from(tickers)

    # 2. Capital Layer (Vectorized Jaccard Similarity)
    n_tickers = len(tickers)
    cap_matrix = np.zeros((n_tickers, n_tickers))
    if not df_holders.empty:
        df_h = df_holders[df_holders['ticker'].isin(tickers)]
        if not df_h.empty:
            pivot_h = pd.crosstab(df_h['ticker'], df_h['Holder'])
            pivot_h = pivot_h.reindex(index=tickers, fill_value=0)
            M_h = pivot_h.values
            intersection = M_h.dot(M_h.T)
            sizes = M_h.sum(axis=1)
            union = sizes[:, None] + sizes[None, :] - intersection
            with np.errstate(divide='ignore', invalid='ignore'):
                cap_matrix = np.where(union > 0, intersection / union, 0.0)

    # 3. Human Layer (Vectorized Shared Insiders)
    human_matrix = np.zeros((n_tickers, n_tickers))
    if not df_insider.empty:
        df_i = df_insider[df_insider['ticker'].isin(tickers)]
        if not df_i.empty:
            pivot_i = pd.crosstab(df_i['ticker'], df_i['owner'])
            pivot_i = pivot_i.reindex(index=tickers, fill_value=0)
            M_i = pivot_i.values
            intersection_i = M_i.dot(M_i.T)
            human_matrix = (intersection_i > 0).astype(float)

    # Fusion Weight Matrix
    fusion_matrix = (alpha * sem_matrix) + (beta * cap_matrix) + (gamma * human_matrix)
    
    # Extract upper triangle indices where fusion > threshold
    rows, cols = np.where(np.triu(fusion_matrix > threshold, k=1))
    edge_count = len(rows)
    
    for i, j in zip(rows, cols):
        G.add_edge(
            tickers[i], tickers[j],
            weight=float(fusion_matrix[i, j]),
            semantic=float(sem_matrix[i, j]),
            capital=float(cap_matrix[i, j]),
            human=float(human_matrix[i, j])
        )
                
    logging.info(f"Graph constructed: {G.number_of_nodes()} nodes, {edge_count} strategic edges.")
    return G

def main():
    config = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = repo_root / "data" / "processed"
    raw_dir = repo_root / "data" / "raw"
    graph_out_dir = repo_root / "data" / "graph_models"

    graph_out_dir.mkdir(parents=True, exist_ok=True)

    # Load Data Sources
    embed_path = processed_dir / "sec_vector_embeddings.parquet"
    holders_path = raw_dir / "institutional_holders.parquet"
    insider_path = raw_dir / "insider_transactions.parquet"

    if not embed_path.exists():
        logging.error("Missing vector embeddings. Run src/sec_parser.py first.")
        return

    df_embed = pd.read_parquet(embed_path)
    df_holders = pd.read_parquet(holders_path) if holders_path.exists() else pd.DataFrame()
    df_insider = pd.read_parquet(insider_path) if insider_path.exists() else pd.DataFrame()

    # Build Unified Graph
    G = build_multi_layer_graph(
        df_embed, df_holders, df_insider,
        threshold=config.get('graph', {}).get('similarity_threshold', 0.6)
    )

    if G.number_of_edges() > 0:
        # Use PageRank and Eigenvector Centrality
        pagerank = nx.pagerank(G, weight="weight")
        eigen_cent = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")

        # Save Graph
        out_path = graph_out_dir / "strategic_network.gml"
        nx.write_gml(G, str(out_path))
        logging.info("Saved multi-layer graph to %s", out_path)

        # Merge Centrality Features
        centrality_df = pd.DataFrame([
            {"ticker": t, "network_pagerank": pagerank[t], "network_eigenvector": eigen_cent[t]}
            for t in G.nodes()
        ])
        
        cent_path = processed_dir / "graph_centrality.parquet"
        centrality_df.to_parquet(cent_path, index=False)
        logging.info("Saved enhanced graph centrality features to %s", cent_path)
    else:
        logging.warning("Graph has no edges. Consider lowering the threshold.")

if __name__ == "__main__":
    main()