"""
Answer Generation Harness with Controlled Prompting and Rate-Limit Resilience.
Calls the 3 competing LLMs (Qwen 2.5, Llama 3.3, Gemini 1.5 Flash)
with identical retrieved context passages under greedy decoding (temp=0.0).
"""

import re
import time
import random
from typing import List, Dict, Any, Optional
import json

from config import (
    GENERATOR_MODELS,
    GROQ_API_KEY,
    GEMINI_API_KEY,
    GENERATOR_TEMPERATURE,
    GENERATOR_MAX_TOKENS,
    CHECKPOINT_FILE,
)

SYSTEM_PROMPT = (
    "You are an expert scientific researcher. Answer the question thoroughly, precisely, and objectively "
    "using ONLY the provided context passages. "
    "Cite specific numerical metrics (accuracies, sensitivities, precision, sample sizes), algorithm names, "
    "feature reduction details, and methodologies explicitly documented in the passages. "
    "If the provided context does not contain sufficient information to answer the question, state: "
    "'The provided context does not contain sufficient information.' "
    "Do not extrapolate or assume facts beyond the provided text."
)


def clean_llm_response(text: str) -> str:
    """Strips internal reasoning / <think> tags if returned by reasoning models."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # If the response was entirely inside <think>, extract the thought text rather than returning empty
    if not cleaned and "<think>" in text:
        match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
    return cleaned if cleaned else text.strip()


class LLMClientManager:
    """Manages API connections, prompts, and robust retry logic for all models."""

    def __init__(self):
        self._groq_client = None
        self._gemini_client = None

    def get_groq_client(self):
        if self._groq_client is None:
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is missing from environment or .env file.")
            from groq import Groq
            self._groq_client = Groq(api_key=GROQ_API_KEY)
        return self._groq_client

    def get_gemini_client(self):
        if self._gemini_client is None:
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is missing from environment or .env file.")
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception:
                import google.generativeai as genai_old
                genai_old.configure(api_key=GEMINI_API_KEY)
                self._gemini_client = genai_old
        return self._gemini_client

    def call_with_retry(self, func, max_retries: int = 4, initial_delay: float = 2.0):
        """Executes API call with exponential backoff and jitter for 429/503 resilience."""
        delay = initial_delay
        last_error = None
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                err_str = str(e).lower()
                last_error = e
                # Check for rate limits or server errors
                if any(x in err_str for x in ["429", "rate limit", "quota", "503", "overloaded", "resource_exhausted"]):
                    sleep_time = delay + random.uniform(0.5, 1.5)
                    print(f"   [API Backoff] Rate limit / quota hit. Retrying in {sleep_time:.2f}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(sleep_time)
                    delay *= 2
                else:
                    raise e
        raise last_error

    def generate_answer(
        self,
        model_key: str,
        question: str,
        context_passages: List[str]
    ) -> Dict[str, Any]:
        """Generates an answer from the specified model using strictly provided context."""
        cfg = GENERATOR_MODELS[model_key]
        provider = cfg["provider"]
        model_id = cfg["model_id"]

        formatted_context = "\n\n".join([f"[Passage {i+1}]: {p}" for i, p in enumerate(context_passages)])
        user_message = (
            f"Context passages:\n{formatted_context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        start_time = time.time()

        if provider == "groq":
            groq = self.get_groq_client()

            def _call_groq():
                res = groq.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=GENERATOR_TEMPERATURE,
                    max_tokens=GENERATOR_MAX_TOKENS,
                )
                msg = res.choices[0].message
                content = msg.content or ""
                if not content.strip() and hasattr(msg, "reasoning") and msg.reasoning:
                    content = msg.reasoning
                return content

            raw_answer = self.call_with_retry(_call_groq)

        elif provider == "gemini":
            gemini = self.get_gemini_client()

            def _call_gemini():
                # Supports both new google-genai and legacy google.generativeai
                if hasattr(gemini, "models"):
                    res = gemini.models.generate_content(
                        model=model_id,
                        contents=f"{SYSTEM_PROMPT}\n\n{user_message}",
                        config={"temperature": GENERATOR_TEMPERATURE, "max_output_tokens": GENERATOR_MAX_TOKENS}
                    )
                    return res.text
                else:
                    model = gemini.GenerativeModel(
                        model_name=model_id,
                        generation_config={"temperature": GENERATOR_TEMPERATURE, "max_output_tokens": GENERATOR_MAX_TOKENS}
                    )
                    res = model.generate_content(f"{SYSTEM_PROMPT}\n\n{user_message}")
                    return res.text

            raw_answer = self.call_with_retry(_call_gemini)
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

        elapsed = time.time() - start_time
        clean_ans = clean_llm_response(raw_answer)
        if len(clean_ans) < 15 and len(raw_answer.strip()) >= 15:
            clean_ans = raw_answer.strip()

        return {
            "answer": clean_ans,
            "latency_seconds": round(elapsed, 3),
            "status": "SUCCESS" if len(clean_ans) >= 15 else "FAILED: Empty output",
            "model_id": model_id,
            "provider": provider
        }


def save_checkpoint(records: Dict[str, Any], path=CHECKPOINT_FILE):
    """Safely persists generation records, resilient to Windows / OneDrive file locks."""
    for attempt in range(5):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            return
        except PermissionError:
            time.sleep(0.3)
        except Exception as e:
            if attempt == 4:
                print(f"[Warning] Failed to write checkpoint: {e}")


def load_checkpoint(path=CHECKPOINT_FILE) -> Dict[str, Any]:
    """Loads existing experiment records if checkpoint exists."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
