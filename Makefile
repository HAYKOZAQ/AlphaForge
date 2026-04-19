# AI-Augmented Quantitative Market Development Intelligence System
# Research Pipeline Makefile

PYTHON = python
SRC_DIR = src
CONFIG_DIR = config
DATA_DIR = data
REPORTS_DIR = reports

.PHONY: all ingest features train eval app clean test

all: ingest features train eval

# Stage 1: Data Ingestion
ingest:
	$(PYTHON) $(SRC_DIR)/data_ingest.py
	$(PYTHON) $(SRC_DIR)/macro_loader.py
	$(PYTHON) $(SRC_DIR)/text_pipeline.py
	$(PYTHON) $(SRC_DIR)/data_ingest_sp500.py
	$(PYTHON) $(SRC_DIR)/sec_ingest.py

# Stage 2: Processing & Features
features:
	$(PYTHON) $(SRC_DIR)/sec_parser.py
	$(PYTHON) $(SRC_DIR)/graph_builder.py
	$(PYTHON) $(SRC_DIR)/feature_builder.py
	$(PYTHON) $(SRC_DIR)/target_builder.py
	$(PYTHON) $(SRC_DIR)/target_builder_sp500.py

# Stage 3: Modeling
train:
	$(PYTHON) $(SRC_DIR)/train.py
	$(PYTHON) $(SRC_DIR)/train_sp500.py

# Stage 4: Evaluation
eval:
	$(PYTHON) $(SRC_DIR)/backtest.py

# Research Dashboard
app:
	streamlit run app/streamlit_app.py

# Testing
test:
	pytest tests/

# Cleanup
clean:
	rm -rf $(DATA_DIR)/raw/*
	rm -rf $(DATA_DIR)/processed/*
	rm -rf $(DATA_DIR)/graph_models/*
	rm -rf $(REPORTS_DIR)/*.json
	rm -rf $(REPORTS_DIR)/*.csv
	rm -rf $(REPORTS_DIR)/plots/*
	rm -rf models/*.pkl
