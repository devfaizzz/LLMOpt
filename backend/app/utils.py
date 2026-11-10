import os
import re
import math
from typing import Optional
from dotenv import load_dotenv

# Load from project root (.env two levels up from this file's directory)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'), override=True)

# ─── Carbon / Energy constants ────────────────────────────────────────────────
# Based on published estimates for GPU inference (A100/H100 class hardware)
ENERGY_PER_TOKEN_KWH = 0.0000003   # ~0.3 µWh per token (conservative avg)
CARBON_INTENSITY_G_KWH = 400       # global avg grid: 400 g CO₂/kWh
WATER_ML_PER_KWH = 1.8             # approx 1.8 ml water / Wh (data-center cooling)

# ─── Model registry: Free tier models on OpenRouter with verified public benchmarks ─
# Sources:
# - Google DeepMind Gemma Technical Report (google.dev, 2024/2025): MMLU-Pro 82.6%, HumanEval ~82.7%
# - NVIDIA Nemotron Technical Reports (nvidia.com, HuggingFace): MMLU ~87%, MMLU-Pro 83.7%, 12B active
# - OpenAI Open Weights Evaluation (openai.com): MMLU 85.3%, GPQA 71.5%, ~3.6B active
# - Poolside AI Technical Whitepaper (vllm.ai): SWE-bench Verified 70.9%, ~3B active MoE
MODEL_REGISTRY = {
    "google/gemma-4-26b-a4b-it:free": {
        "label": "Gemma 4 26B",
        "architecture": "26B MoE (~4B active)",
        "active_params": "~4B",
        "cost_per_1m_tokens": 0.0,
        "cost_per_1m_input_tokens": 0.0,
        "quality_score": 83,  # backwards-compatible general default
        "quality_scores": {
            "general": 83,   # MMLU-Pro 82.6% (Google DeepMind)
            "code": 83,      # HumanEval ~82.7%
            "reasoning": 81, # GSM8K 84.0%
        },
        "benchmark_citation": "MMLU-Pro: 82.6% | HumanEval: 82.7% (Google DeepMind 2024)",
        "carbon_factor": 1.0,  # Baseline: 4B active params = 1.0
        "latency_factor": 1.0, # Baseline inference speed
        "strengths": ["general", "code", "balanced"],
    },
    "nvidia/nemotron-3-super-120b-a12b:free": {
        "label": "Nemotron 3 Super 120B",
        "architecture": "120B MoE (~12B active)",
        "active_params": "~12B",
        "cost_per_1m_tokens": 0.0,
        "cost_per_1m_input_tokens": 0.0,
        "quality_score": 87,
        "quality_scores": {
            "general": 87,   # MMLU 87.0% / MMLU-Pro 83.7% (NVIDIA 2024)
            "code": 85,      # Agentic code reasoning
            "reasoning": 88, # HelpSteer2 & GSM8k: 91.2%
        },
        "benchmark_citation": "MMLU: 87.0% | GSM8K: 91.2% (NVIDIA HelpSteer2 / Nemotron)",
        "carbon_factor": 3.0,  # 12B active params / 4B baseline = 3.0
        "latency_factor": 2.5, # Larger memory footprint / 12B forward pass
        "strengths": ["complex reasoning", "in-depth analysis"],
    },
    "openai/gpt-oss-20b:free": {
        "label": "GPT OSS 20B",
        "architecture": "20B MoE (~3.6B active)",
        "active_params": "~3.6B",
        "cost_per_1m_tokens": 0.0,
        "cost_per_1m_input_tokens": 0.0,
        "quality_score": 85,
        "quality_scores": {
            "general": 85,   # MMLU: 85.3% (OpenAI open weights release)
            "code": 75,      # GPQA: 71.5%
            "reasoning": 86, # Math & Chain-of-thought evaluations
        },
        "benchmark_citation": "MMLU: 85.3% | GPQA: 71.5% (OpenAI Open Evaluation)",
        "carbon_factor": 0.9,  # ~3.6B active / 4B baseline = 0.9
        "latency_factor": 0.95,
        "strengths": ["high quality", "general reasoning"],
    },
    "nvidia/nemotron-nano-12b-v2-vl:free": {
        "label": "Nemotron Nano 12B VL",
        "architecture": "12B Dense SLM",
        "active_params": "12B (Dense)",
        "cost_per_1m_tokens": 0.0,
        "cost_per_1m_input_tokens": 0.0,
        "quality_score": 68,
        "quality_scores": {
            "general": 68,   # MMLU: 78.2%, MMLU-Pro: 64.0% (NVIDIA HF Card)
            "code": 61,      # HumanEval+: 61.0%
            "reasoning": 65, # Edge reasoning
        },
        "benchmark_citation": "MMLU-Pro: 64.0% | HumanEval+: 61.0% (NVIDIA Model Card)",
        "carbon_factor": 2.0,  # 12B dense parameter forward pass
        "latency_factor": 1.4,
        "strengths": ["multimodal", "edge deployment"],
    },
    "poolside/laguna-xs-2.1:free": {
        "label": "Laguna XS 2.1 (Code)",
        "architecture": "33B MoE (~3B active)",
        "active_params": "~3B",
        "cost_per_1m_tokens": 0.0,
        "cost_per_1m_input_tokens": 0.0,
        "quality_score": 71,
        "quality_scores": {
            "general": 58,   # Specialized for code; general reasoning estimated
            "code": 82,      # SWE-bench Verified: 70.9% (high code specialization)
            "reasoning": 55, # Code-centric logic
        },
        "benchmark_citation": "SWE-bench Verified: 70.9% (Poolside AI / vLLM.ai)",
        "carbon_factor": 0.75, # ~3B active / 4B baseline = 0.75 (lowest carbon footprint)
        "latency_factor": 0.75, # Lightweight active execution
        "strengths": ["coding", "ultra-low carbon footprint"],
    },
}

# The optimizer model must be able to handle prompt rewriting
OPTIMIZER_MODEL = "google/gemma-4-26b-a4b-it:free"

# Fallback chain: if the primary optimizer fails, try these in order
OPTIMIZER_FALLBACK_CHAIN = [
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-20b:free",
    "openrouter/free",
]

DEFAULT_MODEL = "google/gemma-4-26b-a4b-it:free"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def count_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token (standard empirical heuristic)."""
    return max(1, math.ceil(len(text) / 4))


def estimate_carbon_metrics(tokens: int, model_key: str = DEFAULT_MODEL) -> dict:
    model = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY[DEFAULT_MODEL])
    cf = model["carbon_factor"]

    energy_kwh = tokens * ENERGY_PER_TOKEN_KWH * cf
    co2_kg = energy_kwh * CARBON_INTENSITY_G_KWH / 1000
    water_ml = energy_kwh * 1000 * WATER_ML_PER_KWH

    usd_cost = (tokens / 1_000_000) * model.get("cost_per_1m_tokens", 0.0)
    inr_cost = usd_cost * 83.5

    return {
        "tokens": tokens,
        "energy_kwh": round(energy_kwh, 8),
        "energy_wh": round(energy_kwh * 1000, 6),
        "co2_kg": round(co2_kg, 8),
        "co2_g": round(co2_kg * 1000, 5),
        "water_ml": round(water_ml, 4),
        "cost_usd": round(usd_cost, 6),
        "cost_inr": round(inr_cost, 4),
    }


def analyze_prompt_issues(prompt: str) -> list[str]:
    issues = []
    words = prompt.split()

    # Repeated phrases
    phrase_counts: dict = {}
    for i in range(len(words) - 2):
        phrase = " ".join(words[i:i+3]).lower()
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
    repeats = [p for p, c in phrase_counts.items() if c > 1]
    if repeats:
        issues.append(f"Repeated phrases detected ({len(repeats)} instance(s))")

    # Filler words
    fillers = ["please", "kindly", "could you", "i want you to", "i need you to",
               "as an ai", "sure", "certainly", "of course", "absolutely"]
    found_fillers = [f for f in fillers if f in prompt.lower()]
    if found_fillers:
        issues.append(f"Filler/polite words that waste tokens: {', '.join(found_fillers[:3])}")

    # Very long prompt
    if len(words) > 200:
        issues.append("Prompt is very long (>200 words) — consider chunking into smaller requests")

    # Redundant enumeration
    redundant_list = re.findall(r'\b(explain|describe|tell me|give me)\b', prompt.lower())
    if len(redundant_list) > 3:
        issues.append(f"Multiple 'explain/describe/tell me' requests ({len(redundant_list)}) — merge them")

    # Verbosity check
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    if avg_word_len < 3.5 and len(words) > 30:
        issues.append("High proportion of short filler words — prompt may be verbose")

    if not issues:
        issues.append("No major issues detected — prompt looks reasonably concise")

    return issues


def detect_task_type(prompt: str) -> str:
    """
    Detect whether prompt is code, reasoning, or general task.
    Returns: 'code' | 'reasoning' | 'general'
    """
    prompt_lower = prompt.lower()
    code_kw = [
        "code", "function", "debug", "implement", "algorithm",
        "python", "javascript", "typescript", "sql", "class", 
        "bug", "error", "compile", "syntax", "program", "method",
        "api", "endpoint", "database", "html", "css", "react"
    ]
    reasoning_kw = [
        "prove", "analyze", "reason", "evaluate", "compare",
        "step by step", "mathematical", "proof", "derive",
        "logic", "argument", "conclusion", "hypothesis", "solve"
    ]

    code_hits = sum(1 for kw in code_kw if re.search(r'\b' + re.escape(kw) + r'\b', prompt_lower))
    reason_hits = sum(1 for kw in reasoning_kw if re.search(r'\b' + re.escape(kw) + r'\b', prompt_lower))

    if code_hits >= 2 or (code_hits >= 1 and ("```" in prompt or "def " in prompt or "function" in prompt)):
        return "code"
    if reason_hits >= 2:
        return "reasoning"
    return "general"


def detect_task_complexity(prompt: str) -> str:
    words = prompt.lower().split()
    complex_keywords = ["analyze", "compare", "evaluate", "reason", "explain in depth",
                        "write code", "debug", "mathematical", "proof", "step by step",
                        "research", "comprehensive", "detailed", "algorithm"]
    simple_keywords = ["what is", "define", "translate", "convert", "list", "name"]

    complex_hits = sum(1 for kw in complex_keywords if kw in prompt.lower())
    simple_hits = sum(1 for kw in simple_keywords if kw in prompt.lower())

    if complex_hits >= 2 or len(words) > 100:
        return "complex"
    elif simple_hits >= 2 or len(words) < 20:
        return "simple"
    return "medium"


def score_models(
    tokens: int,
    task_complexity: str = "medium",
    task_type: str = "general",
    priority: str = "balanced"
) -> dict:
    """
    Multi-Criteria Decision Analysis (MCDA) Scoring Algorithm.
    Computes a composite score [0-100] for each model in MODEL_REGISTRY.
    """
    WEIGHTS = {
        "quality":  {"quality": 0.65, "carbon": 0.15, "cost": 0.10, "latency": 0.10},
        "carbon":   {"quality": 0.20, "carbon": 0.55, "cost": 0.15, "latency": 0.10},
        "cost":     {"quality": 0.20, "carbon": 0.15, "cost": 0.55, "latency": 0.10},
        "balanced": {"quality": 0.40, "carbon": 0.30, "cost": 0.15, "latency": 0.15},
    }
    w = WEIGHTS.get(priority, WEIGHTS["balanced"])

    all_cf = [m["carbon_factor"] for m in MODEL_REGISTRY.values()]
    all_lf = [m["latency_factor"] for m in MODEL_REGISTRY.values()]
    cf_min, cf_max = min(all_cf), max(all_cf)
    lf_min, lf_max = min(all_lf), max(all_lf)

    results = {}
    for key, model in MODEL_REGISTRY.items():
        # 1. Quality score tailored for task type
        q_raw = model.get("quality_scores", {}).get(task_type, model["quality_score"])

        # 2. Carbon score: lower cf -> higher score (0 to 100)
        c_score = 100.0 * (cf_max - model["carbon_factor"]) / (cf_max - cf_min) if cf_max != cf_min else 100.0

        # 3. Latency score: lower latency -> higher score (0 to 100)
        l_score = 100.0 * (lf_max - model["latency_factor"]) / (lf_max - lf_min) if lf_max != lf_min else 100.0

        # 4. Cost score: (currently 100.0 since free tier, scales when commercial pricing applied)
        co_score = 100.0

        # 5. Token penalty: long prompts on high-latency models receive penalty
        penalty = 0.0
        if tokens > 200 and model["latency_factor"] > 1.5:
            penalty = min(15.0, (tokens - 200) / 40.0)

        composite = (
            w["quality"] * q_raw +
            w["carbon"] * c_score +
            w["cost"] * co_score +
            w["latency"] * l_score -
            penalty
        )

        results[key] = {
            "composite_score": round(max(0.0, min(100.0, composite)), 1),
            "quality_component": round(q_raw, 1),
            "carbon_component": round(c_score, 1),
            "latency_component": round(l_score, 1),
            "cost_component": round(co_score, 1),
            "penalty": round(penalty, 1),
        }

    return results


def recommend_model(
    tokens: int,
    task_complexity: str = "medium",
    priority: str = "balanced",
    task_type: str = "general"
) -> str:
    """
    Returns the key of the highest scoring model based on the MCDA scoring engine.
    """
    scores = score_models(tokens, task_complexity, task_type, priority)
    best_key = max(scores, key=lambda k: scores[k]["composite_score"])
    return best_key


def generate_model_comparisons(
    original_tokens: int,
    optimized_tokens: int,
    task_complexity: str = "medium",
    task_type: str = "general",
    priority: str = "balanced"
) -> list[dict]:
    """
    Generates a full comparative footprint (before vs after optimization)
    and algorithmic score breakdown for all models in the registry.
    """
    scores = score_models(original_tokens, task_complexity, task_type, priority)
    best_key = max(scores, key=lambda k: scores[k]["composite_score"])

    comparisons = []
    for key, model in MODEL_REGISTRY.items():
        before_m = estimate_carbon_metrics(original_tokens, key)
        after_m = estimate_carbon_metrics(optimized_tokens, key)

        is_winner = (key == best_key)
        sc = scores[key]

        reason = (
            f"Best overall composite score ({sc['composite_score']}/100) under '{priority}' priority "
            f"for {task_type} tasks." if is_winner else
            f"Scored {sc['composite_score']}/100 with quality {sc['quality_component']}/100 "
            f"and carbon factor {model['carbon_factor']}x."
        )

        comparisons.append({
            "model_key": key,
            "label": model["label"],
            "architecture": model.get("architecture", "Transformer"),
            "active_params": model.get("active_params", "N/A"),
            "benchmark_citation": model.get("benchmark_citation", "Public Benchmark"),
            "quality_score": sc["quality_component"],
            "composite_score": sc["composite_score"],
            "is_recommended": is_winner,
            "selection_reason": reason,
            "carbon_factor": model["carbon_factor"],
            "score_breakdown": sc,
            "before": {
                "tokens": before_m["tokens"],
                "energy_wh": before_m["energy_wh"],
                "co2_g": before_m["co2_g"],
                "water_ml": before_m["water_ml"],
                "cost_inr": before_m["cost_inr"],
                "cost_usd": before_m["cost_usd"],
            },
            "after": {
                "tokens": after_m["tokens"],
                "energy_wh": after_m["energy_wh"],
                "co2_g": after_m["co2_g"],
                "water_ml": after_m["water_ml"],
                "cost_inr": after_m["cost_inr"],
                "cost_usd": after_m["cost_usd"],
            },
            "savings": {
                "energy_wh": round(before_m["energy_wh"] - after_m["energy_wh"], 6),
                "co2_g": round(before_m["co2_g"] - after_m["co2_g"], 5),
                "water_ml": round(before_m["water_ml"] - after_m["water_ml"], 4),
                "pct_reduction": round((1 - optimized_tokens / max(original_tokens, 1)) * 100, 1)
            }
        })

    # Sort so recommended model is first, then descending by composite_score
    comparisons.sort(key=lambda x: (x["is_recommended"], x["composite_score"]), reverse=True)
    return comparisons

