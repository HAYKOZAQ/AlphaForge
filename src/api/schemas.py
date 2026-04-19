from pydantic import BaseModel, Field
from typing import Dict, Any, List

class SectorPredictionRequest(BaseModel):
    date: str = Field(..., description="The date for the prediction features in YYYY-MM-DD format")
    features: Dict[str, float] = Field(..., description="A dictionary of feature names and their values")

class SectorPredictionResponse(BaseModel):
    model_version: str = Field(..., description="The model version used for inference")
    predictions: List[Dict[str, Any]] = Field(..., description="A list containing the prediction scores")
    status: str = Field(default="success", description="Status of the API request")
