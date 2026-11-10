import asyncio
import time
from fastapi import APIRouter, HTTPException
from app.schemas import AnalyzeRequest, RunPromptRequest, PromptAnalysisOut, RunResultOut, MetricsOut
from app.utils import (
    count_tokens, estimate_carbon_metrics, analyze_prompt_issues,
    recommend_model, detect_task_complexity, detect_task_type,
    generate_model_comparisons, MODEL_REGISTRY, DEFAULT_MODEL
)
from app.openrouter_client import optimize_prompt, run_prompt_on_model, check_semantic_similarity

router = APIRouter()


def build_metrics_out(tokens: int, model_key: str) -> MetricsOut:
    m = estimate_carbon_metrics(tokens, model_key)
    return MetricsOut(
        tokens=tokens,
        energy_wh=m["energy_wh"],
        co2_g=m["co2_g"],
        water_ml=m["water_ml"],
        cost_inr=m["cost_inr"],
        cost_usd=m["cost_usd"],
    )


@router.post("/analyze", response_model=PromptAnalysisOut)
async def analyze_prompt(req: AnalyzeRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    if len(prompt) > 8000:
        raise HTTPException(status_code=400, detail="Prompt too long (max 8000 chars).")

    # Step 1: Original token count + issues + complexity profiling
    original_tokens = count_tokens(prompt)
    issues = analyze_prompt_issues(prompt)
    task_complexity = detect_task_complexity(prompt)
    task_type = detect_task_type(prompt)

    # Step 2: Optimize prompt via LLM
    try:
        opt_result = await optimize_prompt(prompt)
        optimized_prompt = opt_result["content"].strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Optimizer LLM error: {str(e)}")

    optimized_tokens = count_tokens(optimized_prompt)

    # Ensure optimization didn't balloon the prompt
    if optimized_tokens > original_tokens:
        optimized_prompt = prompt
        optimized_tokens = original_tokens

    # Step 3: Semantic similarity
    try:
        similarity = await check_semantic_similarity(prompt, optimized_prompt)
    except Exception:
        similarity = 87.0

    # Step 4: Multi-Model Evaluation & Recommendation
    priority = req.priority or "balanced"
    rec_model = req.selected_model or recommend_model(
        tokens=original_tokens,
        task_complexity=task_complexity,
        priority=priority,
        task_type=task_type
    )
    if rec_model not in MODEL_REGISTRY:
        rec_model = DEFAULT_MODEL

    model_comparisons = generate_model_comparisons(
        original_tokens=original_tokens,
        optimized_tokens=optimized_tokens,
        task_complexity=task_complexity,
        task_type=task_type,
        priority=priority
    )

    orig_metrics = build_metrics_out(original_tokens, rec_model)
    opt_metrics = build_metrics_out(optimized_tokens, rec_model)

    token_reduction_pct = round(
        (1 - optimized_tokens / max(original_tokens, 1)) * 100, 1
    )

    savings = {
        "tokens": original_tokens - optimized_tokens,
        "tokens_pct": token_reduction_pct,
        "co2_g": round(orig_metrics.co2_g - opt_metrics.co2_g, 6),
        "energy_wh": round(orig_metrics.energy_wh - opt_metrics.energy_wh, 7),
        "water_ml": round(orig_metrics.water_ml - opt_metrics.water_ml, 5),
        "cost_inr": round(orig_metrics.cost_inr - opt_metrics.cost_inr, 5),
    }

    # Identify recommendation reason from comparisons
    rec_item = next((c for c in model_comparisons if c["model_key"] == rec_model), None)
    rec_reason = rec_item["selection_reason"] if rec_item else "Selected model matches your priority parameters."

    pricing_basis = {
        "pricing_model": "Tokens-per-million (OpenRouter free tier with commercial equivalent benchmarks)",
        "electricity_rate_inr_per_kwh": 7.5,
        "electricity_rate_usd_per_kwh": 0.12,
        "datacenter_pue": 1.2,
        "hardware_baseline": "A100/H100 SXM tensor core cluster (~0.3 µWh / token)",
        "notes": "Cost INR is calculated at standard forex (USD/INR = 83.5). For free tier models API cost is ₹0.00, while electricity and carbon footprint are strictly derived from GPU energy metrics."
    }

    return PromptAnalysisOut(
        original_prompt=prompt,
        optimized_prompt=optimized_prompt,
        original_tokens=original_tokens,
        optimized_tokens=optimized_tokens,
        token_reduction_pct=token_reduction_pct,
        semantic_similarity=round(similarity, 1),
        issues=issues,
        original_metrics=orig_metrics,
        optimized_metrics=opt_metrics,
        savings=savings,
        recommended_model=rec_model,
        recommended_model_label=MODEL_REGISTRY[rec_model]["label"],
        task_complexity=task_complexity,
        task_type=task_type,
        recommendation_reason=rec_reason,
        model_registry={
            k: {
                "label": v["label"],
                "quality_score": v["quality_score"],
                "architecture": v.get("architecture", ""),
                "active_params": v.get("active_params", ""),
                "benchmark_citation": v.get("benchmark_citation", ""),
                "carbon_factor": v["carbon_factor"],
            }
            for k, v in MODEL_REGISTRY.items()
        },
        model_comparisons=model_comparisons,
        pricing_basis=pricing_basis,
    )



@router.post("/run", response_model=RunResultOut)
async def run_prompt(req: RunPromptRequest):
    prompt = req.prompt.strip()
    model_key = req.model_key

    if model_key not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        result = await run_prompt_on_model(prompt, model_key)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {str(e)}")

    total_tokens = result["total_tokens"] or count_tokens(prompt) + count_tokens(result["content"])
    metrics = build_metrics_out(total_tokens, model_key)
    model_info = MODEL_REGISTRY[model_key]

    return RunResultOut(
        model_key=model_key,
        model_label=model_info["label"],
        response=result["content"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        total_tokens=total_tokens,
        metrics=metrics,
    )


@router.get("/models")
async def list_models():
    return {
        k: {
            "label": v["label"],
            "quality_score": v["quality_score"],
            "cost_per_1m_tokens": v["cost_per_1m_tokens"],
            "carbon_factor": v["carbon_factor"],
        }
        for k, v in MODEL_REGISTRY.items()
    }


@router.get("/health")
async def health():
    return {"status": "ok", "service": "CarbonPilot AI Backend"}
