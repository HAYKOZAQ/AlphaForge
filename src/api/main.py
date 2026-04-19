from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import logging
import os
import mlflow
from src.api.schemas import SectorPredictionRequest, SectorPredictionResponse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="AI Market Development Intelligence API",
    description="Institutional-grade research platform API for market development acceleration.",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set MLflow tracking URI (default to local sqlite if not set)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT}/mlflow.db")
mlflow.set_tracking_uri(MLFLOW_URI)

# Global variables to store loaded models
model_cache = {}

def load_latest_model(model_name="model_C"):
    """Loads the latest run of a specific model from MLflow."""
    try:
        if model_name in model_cache:
            return model_cache[model_name]
            
        logger.info(f"Connecting to MLflow at {MLFLOW_URI} to fetch latest {model_name}...")
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("Sector_Development_Acceleration")
        
        if not experiment:
            raise ValueError("Experiment 'Sector_Development_Acceleration' not found in MLflow.")
            
        # Search for runs that match the model name
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{model_name}'",
            order_by=["attribute.start_time DESC"],
            max_results=1
        )
        
        if not runs:
            raise ValueError(f"No completed runs found for {model_name}.")
            
        latest_run = runs[0]
        run_id = latest_run.info.run_id
        
        # Load model
        model_uri = f"runs:/{run_id}/{model_name}"
        logger.info(f"Loading {model_name} from {model_uri}...")
        model = mlflow.pyfunc.load_model(model_uri)
        
        model_cache[model_name] = {
            "model": model,
            "version": run_id
        }
        
        return model_cache[model_name]
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    """Load model on startup so the API is immediately ready."""
    logger.info("Starting up API and pre-loading MLflow models...")
    load_latest_model("model_C")

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "Market Development API"}

@app.post("/predict/sector", response_model=SectorPredictionResponse)
async def predict_sector(request: SectorPredictionRequest):
    """
    Predicts sector development acceleration using the full alternative data pack (Model C).
    """
    model_data = load_latest_model("model_C")
    
    if not model_data:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model is not available or could not be loaded."
        )
        
    model = model_data["model"]
    version = model_data["version"]
    
    try:
        # Convert incoming JSON dict to DataFrame
        df_input = pd.DataFrame([request.features])
        
        # Predict probabilities
        prediction = model.predict(df_input)
        
        # If output is a single value/array, wrap it in a dictionary
        if isinstance(prediction, pd.DataFrame):
            pred_list = prediction.to_dict(orient="records")
        else:
            # Assuming output is a numpy array of probabilities
            pred_list = [{"probability_of_acceleration": float(prediction[0])}]
            
        return SectorPredictionResponse(
            model_version=f"model_C_run_{version}",
            predictions=pred_list
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )
