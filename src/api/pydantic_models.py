"""Pydantic models for the scoring API."""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Feature values for a single customer",
    )


class PredictionResponse(BaseModel):
    risk_probability: float = Field(..., ge=0.0, le=1.0)
    label: Optional[int] = Field(
        None,
        description="Optional hard classification label",
    )
