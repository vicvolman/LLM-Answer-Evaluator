"""
Q&A Evaluator Core Logic - Production Module
Assignment 11.02 - LLM Applications

This module provides core functions for:
- Question selection from repository
- Answer evaluation using LLM + ROUGE metrics
- Feedback collection with sentiment analysis
"""

import json
import uuid
import os
import random
from datetime import datetime
from typing import Optional

# ROUGE metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

# LLM client
try:
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


# ============================================================
# DATA LOADING
# ============================================================

def load_qa_database(filepath: str = "Q&A_db_practice.json") -> list[dict]:
    """Load the question-answer database from JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# QUESTION SELECTION
# ============================================================

def get_question(strategy: str = "random", qa_db: Optional[list] = None) -> dict:
    """
    Select a question from the repository.

    Args:
        strategy: Selection method ("random" or "sequential")
        qa_db: Pre-loaded Q&A database

    Returns:
        dict with question_id, question, target_answer
    """
    if qa_db is None:
        qa_db = load_qa_database()

    if strategy == "random":
        selected = random.choice(qa_db)
    elif strategy == "sequential":
        selected = qa_db[0]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return {
        "question_id": str(uuid.uuid4()),
        "question": selected["question"],
        "target_answer": selected["answer"]
    }


# ============================================================
# ROUGE METRICS
# ============================================================

def compute_rouge(target: str, answer: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    Returns:
        dict with r1, r2, rl keys
    """
    if not ROUGE_AVAILABLE:
        return {"r1": 0.0, "r2": 0.0, "rl": 0.0}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                       use_stemmer=True)
    scores = scorer.score(target, answer)

    return {
        "r1": round(scores['rouge1'].fmeasure, 3),
        "r2": round(scores['rouge2'].fmeasure, 3),
        "rl": round(scores['rougeL'].fmeasure, 3)
    }


# ============================================================
# LLM EVALUATION
# ============================================================

EVALUATION_PROMPT = """You are an expert AI/ML educator evaluating student answers.

**Question:** {question}

**Target Answer:** {target}

**Student Answer:** {answer}

Evaluate the student's answer on three dimensions:
1. **Correctness**: Are the core concepts accurate?
2. **Completeness**: Does it cover key aspects of the target?
3. **Precision**: Is the terminology and explanation clear?

Respond ONLY with valid JSON (no markdown, no extra text):

{{
  "score_0_100": <integer 0-100>,
  "correctness": "<1-2 sentence assessment>",
  "completeness": "<1-2 sentence assessment>",
  "precision": "<1-2 sentence assessment>",
  "rationale": ["<point 1>", "<point 2>", "<point 3>"]
}}

Scoring guide:
- 90-100: Excellent (accurate, comprehensive, precise)
- 70-89: Good (mostly correct, minor gaps)
- 50-69: Partial (some understanding, significant gaps)
- 0-49: Poor (fundamental errors or missing concepts)

Remember: Return ONLY the JSON object, nothing else."""


def evaluate_with_llm(question: str, target: str, answer: str) -> dict:
    """Use LLM to evaluate answer quality."""
    if not LLM_AVAILABLE:
        return {
            "score_0_100": 50,
            "correctness": "LLM not available",
            "completeness": "Cannot assess",
            "precision": "Fallback mode",
            "rationale": ["LLM client not configured"]
        }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "score_0_100": 50,
            "correctness": "API key missing",
            "completeness": "Set OPENAI_API_KEY",
            "precision": "Cannot evaluate",
            "rationale": ["Environment variable OPENAI_API_KEY required"]
        }

    client = OpenAI(api_key=api_key)
    prompt = EVALUATION_PROMPT.format(question=question, target=target, answer=answer)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise evaluator. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        result_text = response.choices[0].message.content.strip()

        # Clean markdown if present
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.rsplit("```", 1)[0]

        evaluation = json.loads(result_text)

        required = {"score_0_100", "correctness", "completeness", "precision", "rationale"}
        if not required.issubset(evaluation.keys()):
            raise ValueError("Missing required keys")

        return evaluation

    except Exception as e:
        return {
            "score_0_100": 50,
            "correctness": "Evaluation failed",
            "completeness": "System error",
            "precision": "Could not process",
            "rationale": [f"Error: {str(e)}"]
        }


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_answer(
    question: str, 
    target: str, 
    answer: str, 
    *, 
    rouge: bool = True,
    question_id: Optional[str] = None
) -> dict:
    """
    Comprehensive answer evaluation.

    Returns:
        dict with eval_id, question_id, model_judgment, rouge, final_score_0_100, timestamp
    """
    eval_id = str(uuid.uuid4())

    llm_judgment = evaluate_with_llm(question, target, answer)
    rouge_scores = compute_rouge(target, answer) if rouge else {"r1": 0.0, "r2": 0.0, "rl": 0.0}

    rouge_avg = (rouge_scores["r1"] + rouge_scores["r2"] + rouge_scores["rl"]) / 3
    final_score = int(0.7 * llm_judgment["score_0_100"] + 0.3 * rouge_avg * 100)

    return {
        "eval_id": eval_id,
        "question_id": question_id or "unknown",
        "model_judgment": llm_judgment,
        "rouge": rouge_scores,
        "final_score_0_100": final_score,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

SENTIMENT_PROMPT = """Analyze the sentiment of this user feedback comment.

**Comment:** {comment}

Classify the sentiment as one of: positive, negative, or neutral.

Respond ONLY with valid JSON (no markdown):

{{
  "sentiment": "<positive|negative|neutral>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<1 sentence explanation>"
}}"""


def analyze_sentiment_llm(comment: Optional[str]) -> dict:
    """Analyze sentiment using LLM."""
    if not comment:
        return {
            "sentiment": "neutral",
            "confidence": 1.0,
            "reasoning": "No comment provided"
        }

    if not LLM_AVAILABLE:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "reasoning": "LLM not available"
        }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "reasoning": "API key not configured"
        }

    client = OpenAI(api_key=api_key)
    prompt = SENTIMENT_PROMPT.format(comment=comment)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert. Return only JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150
        )

        result_text = response.choices[0].message.content.strip()

        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.rsplit("```", 1)[0]

        sentiment = json.loads(result_text)

        if "sentiment" not in sentiment:
            raise ValueError("Missing sentiment field")

        return sentiment

    except Exception as e:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}"
        }


# ============================================================
# FEEDBACK RECORDING
# ============================================================

FEEDBACK_DB = []


def record_feedback(
    eval_id: str, 
    labels: list[str], 
    comment: Optional[str] = None
) -> dict:
    """
    Record user feedback with LLM sentiment analysis.

    Returns:
        dict with feedback_id, eval_id, labels, comment, sentiment_analysis, timestamp
    """
    feedback_id = str(uuid.uuid4())
    sentiment = analyze_sentiment_llm(comment)

    feedback_entry = {
        "feedback_id": feedback_id,
        "eval_id": eval_id,
        "labels": labels,
        "comment": comment,
        "sentiment_analysis": sentiment,
        "timestamp": datetime.now().isoformat()
    }

    FEEDBACK_DB.append(feedback_entry)
    return feedback_entry

# ============================================================
# DEBUG UTILITIES
# ============================================================

def generate_novice_answer(question: str, target: str) -> str:
    """Generate a simplified answer for testing."""
    first_part = target.split('.')[0] if '.' in target else target[:100]
    templates = [
        f"{first_part}... I think.",
        f"I believe {first_part.lower()}",
        f"It's related to {' '.join(first_part.split()[-5:])}",
        "I'm not sure, but it relates to the concept."
    ]
    return random.choice(templates)
