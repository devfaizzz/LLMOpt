from pydantic import BaseModel, ConfigDict
from typing import Optional


class AnalyzeRequest(BaseModel):
    prompt: str
    priority: Optional[str] = "balanced"  # 'quality' | 'carbon' | 'cost' | 'balanced'
    selected_model: Optional[str] = None   # override recommended model


class RunPromptRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prompt: str
    model_key: str


class MetricsOut(BaseModel):
    tokens: int
    energy_wh: float
    co2_g: float
    water_ml: float
    cost_inr: float
    cost_usd: float


class PromptAnalysisOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    original_prompt: str
    optimized_prompt: str
    original_tokens: int
    optimized_tokens: int
    token_reduction_pct: float
    semantic_similarity: float
    issues: list[str]
    original_metrics: MetricsOut
    optimized_metrics: MetricsOut
    savings: dict
    recommended_model: str
    recommended_model_label: str
    task_complexity: str
    task_type: Optional[str] = "general"
    recommendation_reason: Optional[str] = ""
    model_registry: dict
    model_comparisons: Optional[list] = []
    pricing_basis: Optional[dict] = None


class RunResultOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_key: str
    model_label: str
    response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    metrics: MetricsOut
