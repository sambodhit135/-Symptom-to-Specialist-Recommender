"""
src/api/schemas.py
==================
Pydantic schemas for request validation and API responses.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    disease: str = Field(..., description="Canonical disease name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model prediction probability")
    specialist: str = Field(..., description="Recommended medical specialty")


class PredictRequest(BaseModel):
    symptoms: List[str] = Field(
        ...,
        description="List of symptom names presented by the patient",
        examples=[["itching", "skin_rash", "nodal_skin_eruptions"]],
    )


class PredictResponse(BaseModel):
    request_id: str = Field(..., description="Unique UUID for this prediction request")
    model_version: str = Field(..., description="Production model version")
    predictions: List[PredictionItem] = Field(
        ..., min_length=3, max_length=3, description="Top-3 disease predictions sorted by confidence descending"
    )
    emergency_flag: bool = Field(..., description="True if red-flag critical symptoms were detected")
    emergency_reasons: List[str] = Field(
        default_factory=list, description="Clinical triage rationale if emergency flag is true"
    )
    disclaimer: str = Field(
        default="Suggestion tool, not a medical diagnosis.",
        description="Mandatory clinical decision-support disclaimer",
    )


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., description="The request_id from a previous POST /predict call")
    helpful: bool = Field(..., description="Whether the clinician or user found the recommendation helpful")
    true_disease: Optional[str] = Field(
        default=None, description="Optional confirmed disease label (must be in diseases.json if provided)"
    )


class FeedbackResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "recorded"})
    request_id: str = Field(...)


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model_loaded: bool
    model_version: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_alias: str
    model_type: str
    loaded_from: str
    load_time_seconds: float
    num_features: int
    num_classes: int
    quality_gate_passed: bool
