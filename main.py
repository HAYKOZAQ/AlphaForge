import logging
import time

# Sector Core Scripts
from src import data_ingest
from src import macro_loader
from src import text_pipeline
from src import feature_builder
from src import target_builder
from src import train

# Firm-Level / S&P 500 Sub-system Scripts
from src import data_ingest_sp500
from src import sec_ingest
from src import sec_parser
from src import graph_builder
from src import target_builder_sp500
from src import train_sp500

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run_step(step_name, module):
    logging.info("-" * 50)
    logging.info("STARTING: %s", step_name)
    start = time.time()
    try:
        module.main()
        elapsed = time.time() - start
        logging.info("SUCCESS: %s (Duration: %.2fs)", step_name, elapsed)
    except Exception as e:
        logging.error("FAILED: %s. Error: %s", step_name, str(e))
        raise

def main():
    setup_logging()
    logging.info("=" * 70)
    logging.info("INITIATING AI-AUGMENTED RESEARCH PIPELINE")
    logging.info("=" * 70)
    
    start_time = time.time()

    # 1. Sector Model Core
    run_step("Base Market Data Ingestion", data_ingest)
    run_step("Macroeconomic FRED Loading", macro_loader)
    run_step("Headline/Text Signal Extraction", text_pipeline)
    run_step("Sector Feature Construction", feature_builder)
    run_step("Sector Target Generation", target_builder)
    run_step("Sector Model Training", train)

    # 2. Firm-level proof-of-concept extension
    run_step("S&P 500 Component Data Ingestion", data_ingest_sp500)
    run_step("SEC Filing Download", sec_ingest)
    run_step("SEC Filing Parsing & Embedding Generation", sec_parser)
    run_step("Similarity Graph Construction", graph_builder)
    run_step("Firm-Level Target Generation", target_builder_sp500)
    run_step("Firm-Level Proof-of-Concept Model Training", train_sp500)

    elapsed_minutes = (time.time() - start_time) / 60.0
    logging.info("=" * 70)
    logging.info("PIPELINE COMPLETE. Duration: %.2f minutes.", elapsed_minutes)
    logging.info("=" * 70)

if __name__ == "__main__":
    main()
