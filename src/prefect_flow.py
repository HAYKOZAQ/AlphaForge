import logging
from prefect import task, flow
import time

from src import data_ingest
from src import transcript_ingest
from src import macro_loader
from src import insider_ingest
from src import text_pipeline
from src import feature_builder
from src import target_builder
from src import train

from src import data_ingest_sp500
from src import sec_ingest
from src import sec_parser
from src import institutional_ingest
from src import graph_builder
from src import target_builder_sp500
from src import train_sp500
from src import backtest
from src import causal_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_task(module, name):
    @task(name=name, retries=2, retry_delay_seconds=60)
    def wrapped_task():
        logger.info(f"Executing: {name}")
        module.main()
    return wrapped_task

# 1. Sector-level research tasks
t_data_ingest = create_task(data_ingest, "Sector ETF & Options Ingestion")
t_transcript_ingest = create_task(transcript_ingest, "Earnings Call Transcript Ingestion")
t_macro_loader = create_task(macro_loader, "Macroeconomic FRED Loading")
t_insider_ingest = create_task(insider_ingest, "Insider Activity Ingestion")
t_text_pipeline = create_task(text_pipeline, "Headline/Text Signal Extraction")
t_feature_builder = create_task(feature_builder, "Sector Feature Construction")
t_target_builder = create_task(target_builder, "Sector Target Generation")
t_train = create_task(train, "Sector Model Training")

# 2. Firm-level extension tasks
t_data_ingest_sp500 = create_task(data_ingest_sp500, "S&P 500 Component Data Ingestion")
t_institutional_ingest = create_task(institutional_ingest, "Institutional Data Ingestion")
t_sec_ingest = create_task(sec_ingest, "SEC Filing Download")
t_sec_parser = create_task(sec_parser, "SEC Filing Parsing & Embedding Generation")
t_graph_builder = create_task(graph_builder, "Similarity Graph Construction")
t_target_builder_sp500 = create_task(target_builder_sp500, "Firm-Level Target Generation")
t_train_sp500 = create_task(train_sp500, "Firm-Level Proof-of-Concept Model Training")

# 3. Evaluation & Reporting
t_backtest = create_task(backtest, "Sector Backtest & Metrics Export")
t_causal = create_task(causal_inference, "Causal Inference & Robustness Verification")


@flow(name="AI-Augmented Market Development Pipeline")
def market_intelligence_pipeline():
    logger.info("=" * 70)
    logger.info("Starting AI-Augmented Market Development Research Pipeline")
    logger.info("=" * 70)

    # 1. Sector-level research pipeline
    # Independent ingestion steps run concurrently
    r_data = t_data_ingest.submit()
    r_trans = t_transcript_ingest.submit()
    r_macro = t_macro_loader.submit()
    r_insider = t_insider_ingest.submit()
    r_text = t_text_pipeline.submit()
    
    r_data.wait()
    r_trans.wait()
    r_macro.wait()
    r_insider.wait()
    r_text.wait()
    
    r_feat = t_feature_builder.submit(wait_for=[r_data, r_trans, r_macro, r_insider, r_text])
    r_feat.wait()
    
    r_targ = t_target_builder.submit(wait_for=[r_feat])
    r_targ.wait()
    
    r_train = t_train.submit(wait_for=[r_targ])
    r_train.wait()

    # 2. Firm-level proof-of-concept extension
    r_data_sp500 = t_data_ingest_sp500.submit()
    r_inst = t_institutional_ingest.submit()
    r_sec = t_sec_ingest.submit()
    
    r_data_sp500.wait()
    r_inst.wait()
    r_sec.wait()
    
    r_sec_parse = t_sec_parser.submit(wait_for=[r_sec])
    r_sec_parse.wait()
    
    r_graph = t_graph_builder.submit(wait_for=[r_sec_parse, r_inst])
    r_graph.wait()
    
    r_targ_sp500 = t_target_builder_sp500.submit(wait_for=[r_graph, r_data_sp500])
    r_targ_sp500.wait()
    
    r_train_sp500 = t_train_sp500.submit(wait_for=[r_targ_sp500])
    r_train_sp500.wait()

    # 3. Evaluation & Reporting
    r_backtest = t_backtest.submit(wait_for=[r_train])
    r_causal = t_causal.submit(wait_for=[r_train])
    
    r_backtest.wait()
    r_causal.wait()

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE.")
    logger.info("=" * 70)

if __name__ == "__main__":
    market_intelligence_pipeline()