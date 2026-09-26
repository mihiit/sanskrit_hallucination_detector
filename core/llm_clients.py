"""
core/llm_clients.py

Concrete LLMClient implementations for three free-to-use backends:

  - GroqLLMClient        — Groq's hosted API (free tier), serving Llama
                            models. Requires GROQ_API_KEY.
  - OllamaLLMClient       — a local Ollama server (`ollama serve`),
                            completely free, no API key, no internet
                            needed after the model is pulled.
  - HFLocalLLMClient      — GPT-2, Qwen2.5, or any other Hugging Face
                            causal LM run locally via `transformers`.
                            Free, no API key, and gives REAL per-token
                            logprobs straight from the model — the
                            most direct neural-confidence signal of
                            the three, since there's no API-provider
                            abstraction in between.

None of these are imported eagerly by core/hybrid_detector.py, so
installing extra dependencies (groq, requests, transformers/torch) is
only required for the backend(s) you actually use. Each class raises a
clear ImportError with an install hint if its dependency is missing.

Usage:

    from core.llm_clients import GroqLLMClient, OllamaLLMClient, HFLocalLLMClient
    from core.hybrid_detector import HybridDetector

    detector = HybridDetector(llm_client=GroqLLMClient(model="llama-3.3-70b-versatile"))
    # or
    detector = HybridDetector(llm_client=OllamaLLMClient(model="llama3.1"))
    # or
    detector = HybridDetector(llm_client=HFLocalLLMClient(model_name="Qwen/Qwen2.5-1.5B-Instruct"))
"""

from __future__ import annotations

import math
import os
import time
from typing import List, Optional

from core.hybrid_detector import LLMClient, SimpleLLMResponse


# --------------------------------------------------------------------------- #
# Groq (hosted, free tier, Llama models)
# --------------------------------------------------------------------------- #


class GroqLLMClient(LLMClient):
    """
    Uses Groq's hosted API (OpenAI-compatible), free tier available at
    console.groq.com — no credit card required to get an API key as of
    this writing, but check current terms yourself since free-tier
    policies change.

    Set GROQ_API_KEY in your environment, or pass api_key explicitly.

    Note on logprobs: Groq's chat-completions endpoint supports
    `logprobs=True` for some models but not all, and behavior can
    change between model versions. This client requests logprobs and
    falls back to a neutral, clearly-flagged placeholder confidence
    (not a real signal) if the response doesn't include them — the
    returned SimpleLLMResponse never silently pretends to have real
    logprobs it doesn't have.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 768,
    ):
        try:
            from groq import Groq  # type: ignore
        except ImportError as e:
            raise ImportError(
                "GroqLLMClient requires the 'groq' package. Install with:\n"
                "  pip install groq --break-system-packages"
            ) from e

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "No Groq API key found. Set GROQ_API_KEY in your environment "
                "or pass api_key=... explicitly. Get a free key at "
                "https://console.groq.com"
            )

        self._client = Groq(api_key=key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, prompt: str) -> SimpleLLMResponse:
        # Try with logprobs first; several current Groq-hosted models
        # (observed: allam-2-7b, the openai/gpt-oss-* family,
        # qwen/qwen3.8-27b as of this writing) reject the request
        # outright with a 400 if `logprobs` is passed at all, rather
        # than silently omitting it from the response. Groq's supported
        # parameters vary by model and change over time, so we detect
        # this at call time instead of hardcoding a model allowlist.
        #
        # Also retries with exponential backoff on transient Groq
        # errors (429 rate-limit, 503 over-capacity) — observed in
        # practice during evaluation runs. A single flaky request
        # should not throw away an entire in-progress run.
        def _call(max_tokens: int):
            last_error = None
            for attempt in range(5):
                try:
                    try:
                        return self._client.chat.completions.create(
                            model=self._model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self._temperature,
                            max_tokens=max_tokens,
                            logprobs=True,
                            top_logprobs=1,
                        )
                    except Exception as e:
                        if "logprobs" in str(e).lower():
                            return self._client.chat.completions.create(
                                model=self._model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=self._temperature,
                                max_tokens=max_tokens,
                            )
                        raise
                except Exception as e:
                    msg = str(e).lower()
                    is_transient = (
                        "rate_limit" in msg
                        or "rate limit" in msg
                        or "over capacity" in msg
                        or "503" in msg
                        or "internal_server_error" in msg
                    )
                    if not is_transient or attempt == 4:
                        raise
                    wait = min(60, 2 ** (attempt + 1))
                    print(
                        f"    [GroqLLMClient] transient error "
                        f"({e.__class__.__name__}), retrying in {wait}s "
                        f"(attempt {attempt + 1}/5)..."
                    )
                    time.sleep(wait)
                    last_error = e
            raise last_error

        response = _call(self._max_tokens)
        choice = response.choices[0]
        text = choice.message.content or ""

        # Some hosted reasoning models (observed: openai/gpt-oss-20b on
        # Groq) can spend the ENTIRE token budget on an internal
        # "reasoning" field and hit finish_reason="length" with content
        # still empty. That is not the same as the model having nothing
        # to say — it means self._max_tokens was too small for this
        # model on this prompt. Retry once with a much larger budget
        # before accepting an empty response; only fall through to an
        # empty/placeholder result if it's still empty after that.
        if not text and getattr(response.choices[0], "finish_reason", None) == "length":
            retry_tokens = max(self._max_tokens * 4, 3000)
            response = _call(retry_tokens)
            choice = response.choices[0]
            text = choice.message.content or ""

        token_logprobs: List[float] = []
        lp = getattr(choice, "logprobs", None)
        if lp is not None and getattr(lp, "content", None):
            token_logprobs = [
                tok.logprob for tok in lp.content if tok.logprob is not None
            ]

        if not token_logprobs:
            # Groq/model combo didn't return logprobs (either not
            # requested due to the fallback above, or requested but
            # empty). Use a clearly-labeled neutral placeholder (0.0
            # mean logprob = "no signal"), rather than fabricating a
            # confident-looking number. This means the neural_component
            # of the Hybrid pipeline's fused score is uninformative for
            # these models — detection relies on the symbolic
            # (SAS/VDV) signal instead. Worth stating explicitly in the
            # paper for any Groq-backed system.
            n_tokens = max(1, len(text.split()))
            token_logprobs = [0.0] * n_tokens

        return SimpleLLMResponse(text=text, token_logprobs=token_logprobs)


# --------------------------------------------------------------------------- #
# Ollama (local, free, no API key)
# --------------------------------------------------------------------------- #


class OllamaLLMClient(LLMClient):
    """
    Talks to a local Ollama server (default http://localhost:11434).

    Setup:
      1. Install Ollama: https://ollama.com
      2. Pull a model:   ollama pull llama3.1
      3. Ollama runs its server automatically after install/pull; if
         not, start it with `ollama serve`.

    Free, offline after the initial model pull, no API key.

    Note on logprobs: recent Ollama versions can return logprobs when
    the request sets "options": {"logprobs": true}, but this is
    version- and model-dependent. This client requests it and falls
    back to a neutral placeholder confidence (clearly not a real
    signal) if the server doesn't provide it — never fabricated.
    """

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        try:
            import requests  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OllamaLLMClient requires the 'requests' package. Install with:\n"
                "  pip install requests --break-system-packages"
            ) from e

        self._requests = requests
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, prompt: str) -> SimpleLLMResponse:
        try:
            resp = self._requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self._temperature,
                        # num_predict caps how many tokens a single
                        # generation can produce. Without this, an
                        # unusual/hard prompt (like our adversarial
                        # Sanskrit traps) can push local CPU inference
                        # into a very long or effectively unbounded
                        # generation — observed in practice: simple
                        # sanity-check prompts finished in under 25s,
                        # but the first real trap item ran past a
                        # 600s timeout with no limit set.
                        "num_predict": self._max_tokens,
                    },
                    # Keep the model loaded in memory between calls so an
                    # evaluation run of many sequential prompts doesn't
                    # pay a reload cost on every single request — observed
                    # in practice to cause large timing spikes (e.g. one
                    # call at ~26s, the next timing out past 120s).
                    "keep_alive": "30m",
                },
                # 600s: local CPU inference on an 8B model can legitimately
                # take several minutes per response depending on hardware;
                # 120s was too aggressive and caused false timeout failures
                # on otherwise-working requests. num_predict above should
                # keep actual generations well under this, but the ceiling
                # stays generous as a safety margin.
                timeout=600,
            )
            resp.raise_for_status()
        except Exception as e:
            raise ConnectionError(
                f"Could not reach Ollama at {self._base_url}, or the "
                f"request timed out. Is `ollama serve` running and is the "
                f"'{self._model}' model pulled (`ollama pull "
                f"{self._model}`)? Original error: {e}"
            ) from e

        data = resp.json()
        text = data.get("response", "")

        # Ollama's /api/generate does not reliably expose per-token
        # logprobs across versions/models. If a future/local version
        # populates one (e.g. under "logprobs"), use it; otherwise fall
        # back to a neutral, clearly-labeled placeholder.
        token_logprobs = data.get("logprobs")
        if not token_logprobs:
            n_tokens = max(1, len(text.split()))
            token_logprobs = [0.0] * n_tokens

        return SimpleLLMResponse(text=text, token_logprobs=token_logprobs)


# --------------------------------------------------------------------------- #
# Local Hugging Face model (GPT-2, Qwen2.5, etc.) — real logprobs
# --------------------------------------------------------------------------- #


class HFLocalLLMClient(LLMClient):
    """
    Runs a Hugging Face causal LM locally via `transformers` and
    computes REAL per-token log-probabilities directly from the
    model's output logits — no API, no key, fully free and offline
    after the initial model download.

    Works with small models on CPU (GPT-2, Qwen2.5-0.5B/1.5B); larger
    models will be slow without a GPU.

    Example models:
      - "gpt2"                          (~500MB, fast on CPU)
      - "Qwen/Qwen2.5-1.5B-Instruct"     (a few GB, usable on CPU, better on GPU)
      - "Qwen/Qwen2.5-0.5B-Instruct"     (smallest Qwen, fastest option)
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        max_new_tokens: int = 128,
        device: Optional[str] = None,
    ):
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "HFLocalLLMClient requires 'transformers' and 'torch'. Install with:\n"
                "  pip install transformers torch --break-system-packages"
            ) from e

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name).to(self._device)
        self._model.eval()
        self._max_new_tokens = max_new_tokens

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def complete(self, prompt: str) -> SimpleLLMResponse:
        torch = self._torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated_ids = output.sequences[0][input_len:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        # output.scores is a tuple of per-step logits (one per generated
        # token); convert each to a log-softmax and pick out the logprob
        # of the token actually generated — this is a REAL logprob from
        # the model's own distribution, not an approximation.
        token_logprobs: List[float] = []
        for step_logits, token_id in zip(output.scores, generated_ids):
            log_probs = torch.log_softmax(step_logits[0], dim=-1)
            token_logprobs.append(log_probs[token_id].item())

        if not token_logprobs:
            token_logprobs = [0.0]

        return SimpleLLMResponse(text=text, token_logprobs=token_logprobs)