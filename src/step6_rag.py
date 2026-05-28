"""
Step 6 — Retrieval-Augmented Generation (RAG).

WHAT RAG IS, IN ONE PARAGRAPH
-----------------------------
A small or even a large language model has two limits when you ask it
about something specific (your company docs, a new HPC cluster, last
week's news): it doesn't know facts that weren't in its training data,
and when it doesn't know, it confidently makes things up ("hallucinates").
RAG is the fix. Instead of asking the LLM "answer from your memory",
we first *retrieve* the most relevant snippets from our own document
collection (steps 1-5 of this project) and then *augment* the model's
prompt with those snippets. The LLM's job changes from "remember the
answer" to "read these passages and write the answer". That's a much
easier job, and the answer is grounded in source text we can cite.

ARCHITECTURE (DATA FLOW)
------------------------

    user question
         │
         ▼
   ┌──────────────────┐         ┌────────────────────────┐
   │ embed(query)     │────────▶│ Milvus  (top-K search) │
   │ (MiniLM, 384-d)  │         │ IVF_FLAT + COSINE      │
   └──────────────────┘         └─────────────┬──────────┘
                                              │ top-K chunks
                                              ▼
                                ┌────────────────────────┐
                                │ build prompt:          │
                                │   system + context +   │
                                │   question             │
                                └─────────────┬──────────┘
                                              ▼
                                ┌────────────────────────┐
                                │ Qwen2.5-1.5B-Instruct  │
                                │ (HF transformers)      │
                                └─────────────┬──────────┘
                                              ▼
                                       grounded answer
                                       (+ source links)

Five players: an embedder (MiniLM), a vector DB (Milvus Lite), a prompt
template, an instruction-tuned LLM (Qwen2.5-1.5B), and a tokenizer that
sits between them. Steps 1-4 of this project produced the vector DB.
Step 5 wrote `search()` which does the first two boxes. This file does
the last three.

WHY QWEN2.5-1.5B-INSTRUCT
-------------------------
Picking an LLM for a learning project means trading off three things:

  - quality            (bigger model, better answers)
  - speed              (smaller model, faster answers)
  - "weight" on disk   (you have to download it)

Qwen2.5-1.5B-Instruct hits a sweet spot:
  * 1.5 B parameters, ~3 GB on disk in float32 (we load in bfloat16 on
    GPU/MPS so it's ~1.5 GB in RAM, plain float32 on CPU).
  * Apache-2.0 license — free for any use, no auth wall on HF.
  * One of the most downloaded small *instruct* models on Hugging Face,
    so there's a lot of community knowledge if something breaks.
  * Strong at "answer from the given context" tasks, which is exactly
    what RAG needs. (For RAG the LLM doesn't have to *know* the facts;
    we hand it the facts in the prompt.)

THE PROMPT TEMPLATE
-------------------
The single highest-leverage choice in a RAG system, after retrieval
quality, is how you stitch the retrieved chunks into the prompt. Ours
follows the standard pattern from the Milvus + LangChain tutorials:

    <system> You are a helpful assistant. Answer ONLY from the
             provided context. If the answer isn't there, say so.
    <user>   Context:
             [1] <chunk text>
             [2] <chunk text>
             ...
             Question: <user question>
             Answer:

Why every piece matters:
  * The system instruction tells the model the *rules* — answer from
    context, don't invent. Without this you get more hallucination.
  * Numbering the chunks lets the model (and you) cite "[2]" without
    quoting the whole passage.
  * "Answer:" is a tiny trick that biases the model to start answering
    immediately instead of repeating the question.

We use the tokenizer's `apply_chat_template` so the system/user roles
are wrapped in whatever special tokens Qwen was fine-tuned on — getting
this wrong is the #1 reason RAG answers come out garbled.

GENERATION KNOBS
----------------
  * temperature = 0.2  : low randomness; we want the model to stick to
    the retrieved text, not embellish.
  * max_new_tokens = 350 : caps answer length so generation latency is
    bounded and the model doesn't ramble.
  * device = auto      : CUDA > MPS (Apple Silicon GPU) > CPU. On CPU
    expect 30-60 s per answer; on MPS expect 5-15 s.

HOW TO RUN
----------
    python -m src.step6_rag "how do I submit a multi-node job on Polaris?"

The first invocation downloads ~3 GB of Qwen weights into the Hugging
Face cache (~/.cache/huggingface) — subsequent runs are instant to
load. We `lru_cache` the model so repeated calls inside one process
don't reload it.
"""

from __future__ import annotations

import sys
import time
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    LLM_DEVICE,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    TOP_K,
)
from src.step5_search import search


# ---------------------------------------------------------------------------
# Device selection — pick the fastest backend that actually works.
# ---------------------------------------------------------------------------
def _pick_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' based on what's available.

    Order of preference:
      1. CUDA  — NVIDIA GPU, fastest for transformers.
      2. MPS   — Apple Silicon GPU (M1/M2/M3). 5-10x faster than CPU.
      3. CPU   — always works, slow but fine for a 1.5B model.

    If the user pinned LLM_DEVICE in config.py we respect that.
    """
    if LLM_DEVICE is not None:
        return LLM_DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Model loader — cached so we only pay the load cost once per process.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_llm() -> tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    """Download (first time) + load the LLM and its tokenizer.

    We pick a dtype based on the device:
      * bfloat16 on CUDA/MPS — halves memory, ~no quality loss for
        inference. Modern GPUs have native bf16 paths.
      * float32 on CPU — bf16 on CPU is *slower* in PyTorch today; the
        scalar code path isn't bf16-optimised.

    `device_map=device` puts every weight on the chosen device in one
    shot. For a 1.5B model this is fine; for a 70B model you'd use
    `device_map="auto"` so accelerate splits across multiple GPUs.
    """
    device = _pick_device()
    dtype  = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    print(f"  loading LLM {LLM_MODEL!r} on {device} (dtype={str(dtype).split('.')[-1]})...")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()  # disables dropout etc. — we're doing inference, not training

    print(f"  LLM ready in {time.perf_counter() - t0:.2f}s")
    return tokenizer, model, device


# ---------------------------------------------------------------------------
# Prompt construction — the actual "augmentation" step of RAG.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant for the Argonne Leadership Computing "
    "Facility (ALCF) user documentation. Answer the user's question "
    "using ONLY the numbered context passages provided. If the answer "
    "is not in the context, say you don't know rather than guessing. "
    "Be concise and, when helpful, cite the passage number like [2]."
)


def _build_messages(query: str, hits: list[dict]) -> list[dict]:
    """Stitch the retrieved chunks into a chat-formatted prompt.

    We return a list of {role, content} dicts in the format the Qwen
    tokenizer's `apply_chat_template` expects. Two messages:

      * system : the rules of the game (answer from context only).
      * user   : the numbered context, followed by the question.

    The tokenizer turns this into the exact token sequence Qwen2.5 was
    fine-tuned on (<|im_start|>system ... <|im_end|>, etc.). Building
    the string by hand is the #1 source of garbled RAG answers, so let
    the tokenizer do it.
    """
    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        text = hit.get("entity", {}).get("text", "").strip()
        context_blocks.append(f"[{i}] {text}")
    context = "\n\n".join(context_blocks)

    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]


# ---------------------------------------------------------------------------
# Generation — turn the prompt into tokens, run the model, decode back.
# ---------------------------------------------------------------------------
def _generate(tokenizer, model, device: str, messages: list[dict]) -> str:
    """Run a single forward generation pass and return the decoded answer.

    Steps inside this function map 1:1 to the transformer inference loop:
      1. `apply_chat_template`  : messages -> token IDs (with role tags).
      2. `model.generate`       : autoregressive decoding for up to
                                  LLM_MAX_NEW_TOKENS new tokens.
      3. `tokenizer.decode`     : token IDs back to a Python string.

    We slice off the prompt tokens before decoding so the returned text
    is *only* the answer, not the prompt echoed back.
    """
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,   # tells the template "now it's the model's turn to speak"
        return_tensors="pt",
    ).to(device)

    # `torch.no_grad` disables autograd — we don't need gradients at
    # inference time and skipping them saves memory + a bit of speed.
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            do_sample=LLM_TEMPERATURE > 0,
            temperature=LLM_TEMPERATURE,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # `output_ids` contains [prompt_tokens | generated_tokens]. Slice off
    # the prompt so we decode only the new tokens.
    generated = output_ids[0, input_ids.shape[-1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer


# ---------------------------------------------------------------------------
# Public entry point — the full retrieve-augment-generate pipeline.
# ---------------------------------------------------------------------------
def rag_answer(query: str, top_k: int = TOP_K) -> str:
    """Answer `query` using retrieved ALCF doc chunks + the local LLM.

    Returns the answer string and prints it (plus the cited sources)
    so the script is useful both as a library function and a CLI demo.
    """
    # ---- 1. RETRIEVE -----------------------------------------------------
    # Reuse step 5's search(): it embeds the query, hits Milvus, and
    # prints the top-K hits for us. No duplication of search logic here.
    print("\n  [RAG] retrieving context...")
    hits = search(query, top_k=top_k)
    if not hits:
        msg = "No relevant context found. Cannot answer."
        print(f"\n  ANSWER: {msg}")
        return msg

    # ---- 2. AUGMENT ------------------------------------------------------
    # Build the chat-formatted prompt with the retrieved chunks.
    messages = _build_messages(query, hits)

    # ---- 3. GENERATE -----------------------------------------------------
    tokenizer, model, device = _load_llm()
    print("\n  [RAG] generating answer...")
    t0 = time.perf_counter()
    answer = _generate(tokenizer, model, device, messages)
    t_gen = time.perf_counter() - t0
    print(f"  [RAG] generated in {t_gen:.2f}s ({device})")

    # ---- 4. PRESENT ------------------------------------------------------
    print("\n" + "=" * 64)
    print("  ANSWER")
    print("=" * 64)
    print(f"  {answer}")
    print("\n  SOURCES")
    print("  " + "-" * 60)
    for i, hit in enumerate(hits, start=1):
        entity = hit.get("entity", {})
        print(f"  [{i}] {entity.get('title', '')}")
        print(f"      {entity.get('url', '')}")

    return answer


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "How do I submit a multi-node GPU job on Polaris?"
    rag_answer(user_query)
