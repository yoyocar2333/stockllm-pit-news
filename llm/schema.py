from typing import Literal
from pydantic import BaseModel, Field, field_validator


class NewsSignal(BaseModel):
    sentiment: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    importance: int = Field(ge=1, le=5)
    risk_level: Literal["Low", "Medium", "High"]
    event_type: str = Field(min_length=1, max_length=80)
    time_horizon: Literal["Short", "Medium", "Long"]
    market_relevance: float = Field(ge=0, le=1)
    direction_probability: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=1000)

    @field_validator("event_type")
    @classmethod
    def clean_event(cls, value):
        return value.strip()
