<div align="center">

```
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   $ git merge openai/grok xai-org/grok-1                                 ║
║                                                                          ║
║   CONFLICT (rename): both repos are named "grok"                         ║
║   CONFLICT (content): PyTorch ≠ JAX                                      ║
║   CONFLICT (scale): 100,000 params ≠ 314,000,000,000 params              ║
║   CONFLICT (purpose): studying understanding ≠ claiming to understand     ║
║   CONFLICT (history): OpenAI ≠ xAI                                       ║
║                                                                          ║
║   Automatic merge failed. Proceeding anyway.                             ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

# Does Grok grok grokking?

[`openai/grok`](https://github.com/openai/grok) · [`xai-org/grok-1`](https://github.com/xai-org/grok-1) · bridged

</div>

---

**grok** — three words that happen to be the same word:

| | Meaning | Source |
|---|---|---|
| **Grok** *(noun)* | xAI's 314B-parameter Mixture-of-Experts LLM, open-sourced March 17, 2024 | *xai-org/grok-1* |
| **grok** *(verb)* | to understand something so thoroughly that observer and observed merge | Heinlein, *Stranger in a Strange Land*, 1961 |
| **grokking** *(ML noun)* | a phase transition where neural networks suddenly generalize long after memorizing training data | Power et al., 2022; *openai/grok* |

This repository asks one question with three interpretations: **does the model named Grok deeply understand the phenomenon called grokking?**

And, more concretely: does Grok-1's architecture — when miniaturized and trained on arithmetic — exhibit different grokking dynamics than a standard transformer?

---

## Background

In February 2023, Elon Musk publicly accused OpenAI of betraying its founding mission as an open-source nonprofit. In November 2023, xAI launched **Grok** as a closed product. On February 29, 2024, Musk filed a lawsuit against OpenAI. Eleven days later, he announced Grok would be open-sourced — and released it on March 17.

Somewhere in that timeline, two GitHub repositories existed under the name `grok`:

```
github.com/openai/grok       ←  ~500 lines, PyTorch, studying when models learn
github.com/xai-org/grok-1    ←  ~1400 lines, JAX, a model claiming it has learned
```

No one had thought to connect them. This fork does.

---

## Architecture

```
   openai/grok (original)              this fork adds              xai-org/grok-1 (original)
   ─────────────────────────                                       ──────────────────────────

   ┌───────────────────────┐                                       ┌──────────────────────────┐
   │  Dense Transformer    │                                       │  Mixture-of-Experts      │
   │                       │                                       │                          │
   │  Input Embedding      │                                       │  Input Embedding         │
   │  + sinusoidal PE      │ ─── replicated as RoPE ────────────► │  + Rotary PE (RoPE)      │
   │                       │                                       │                          │
   │  Multi-Head Attention │ ─── replicated as GQA ─────────────► │  GQA (48q / 8kv heads)   │
   │                       │                                       │                          │
   │  FFN                  │                                       │  MoE FFN                 │
   │  ReLU(xW₁)W₂          │ ─── replicated as MoE + gating ────► │  8 experts, top-2        │
   │                       │                                       │  GELU(xW₁) ⊙ (xWᵥ) W₂  │
   │  LayerNorm            │ ─── replicated as RMSNorm ──────────► │  RMSNorm                 │
   └───────────────────────┘                                       └──────────────────────────┘

        ~100K parameters                                               314,000,000K parameters
        PyTorch + Lightning                                            JAX + Haiku
        task: 42 + 55 mod 97                                          task: everything
```

The resulting class — `GrokOneTransformer` in `grok/transformer.py` — is a PyTorch implementation of Grok-1's decoder architecture that plugs directly into OpenAI's training and evaluation framework. The same training loop, the same arithmetic datasets, a fundamentally different optimizer landscape.

---

## Bridges

Three concrete connections were made between the two codebases:

**Bridge A — Architecture port (JAX → PyTorch)**

`GrokOneTransformer` brings Grok-1's full architectural stack into the grokking framework. Train it against the standard transformer on identical tasks to study whether MoE changes the memorization-to-generalization phase transition.

```bash
# Standard transformer (original)
./grok-main/scripts/train.py --math_operator + --train_data_pct 5

# Grok-1 architecture, same task
./grok-main/scripts/train.py --architecture grok1 --math_operator + --num_experts 8

# Auto-scaled miniature Grok-1
./grok-main/scripts/train.py --architecture grok1_mini
```

New metrics track routing-specific grokking signals (routing entropy, expert specialization, collapse index) logged per epoch alongside the standard weight norm and generalization bounds.

**Bridge B — Arithmetic evaluation (OpenAI tasks → Grok-1 inference)**

`run.py` in `grok-1-main/` gains an `--eval-grokking` flag that generates modular arithmetic problems in the style of the OpenAI paper and scores Grok-1's responses.

```bash
# Dry run — inspect problem generation without a checkpoint
python grok-1-main/run.py --eval-grokking --operator + --n-samples 50 --dry-run

# Full evaluation (requires the 314B checkpoint, ~300GB)
python grok-1-main/run.py --eval-grokking --operator + --n-samples 100
```

**Bridge C — Config export**

`TransformerConfig` in `grok-1-main/model.py` can now export itself as a scaled-down PyTorch-compatible config:

```python
from grok_1_main.model import TransformerConfig

full_config = TransformerConfig(
    emb_size=6144, num_layers=64, num_q_heads=48,
    num_kv_heads=8, num_experts=8, num_selected_experts=2,
    widening_factor=8,
)

# Scale down by 1/24 for a trainable experiment
mini = full_config.to_grokking_config(scale_factor=1/24)
# → {'d_model': 256, 'n_layers': 2, 'n_heads': 2, 'num_experts': 8, ...}
```

---

## The Research Question

Beyond the provocation, there is a real empirical question here.

The 2022 grokking paper studied dense transformers on algorithmic tasks and found a universal pattern: models memorize first, then — often thousands of steps later — abruptly generalize. The phase transition is sharp, nearly discontinuous, and poorly understood.

Mixture-of-Experts models have a fundamentally different optimization geometry. The router introduces a discrete, non-differentiable dispatch decision at each layer. This creates a non-smooth loss landscape, load-balancing pressures, and the possibility of "routing collapse" — where one expert handles all inputs.

Three testable hypotheses:

1. **Routing entropy as a leading indicator** — Does the distribution over experts become more uniform *before* the grokking transition appears in the loss curve? If so, routing entropy might predict generalization before it happens.

2. **Expert collapse → memorization trap** — If all tokens route to one expert during memorization, that expert overfits. Grokking might require the router to "spread out" first.

3. **MoE capacity delays grokking onset** — More parameters means more room to memorize without generalizing. Does a larger expert count push the phase transition further out in training time?

None of these have been tested. This fork provides the infrastructure to test them.

---

## Quick Start

```bash
# Verify both architectures instantiate and run forward passes
python does_grok_grok.py --demo

# Run a comparative grokking experiment (logs to ./logs/)
python does_grok_grok.py --experiment --operator + --max-steps 50000

# Test Grok-1's arithmetic ability (no checkpoint needed for dry run)
python does_grok_grok.py --eval-grok1 --dry-run --n-samples 20
```

Expected output from `--demo`:

```
[1] Standard transformer (dense, sinusoidal PE)
    Params: 329,860  ·  2L / 4H / 128D

[2] Grok-1 architecture (MoE + RoPE + RMSNorm + gated GELU)
    Params: 4,610,148  ·  2L / 2H / 256D / 8 experts (top-2)

[3] MoE routing (layer 0):
    Expert 0  ████████░░░░░░░░  0.125
    Expert 1  ████████░░░░░░░░  0.125
    Expert 2  ████████░░░░░░░░  0.125
    ...

    Routing entropy: 2.079 / 2.079 (perfectly uniform)

[4] Config export (Grok-1 → PyTorch scale):
    {'d_model': 256, 'n_layers': 2, 'n_heads': 2, 'num_experts': 8, ...}

Both architectures operational. Bridge verified.
```

---

## Structure

```
.
├── does_grok_grok.py           unified entry point
├── README.md                   you are here
│
├── grok-main/                  fork of openai/grok
│   ├── grok/
│   │   ├── transformer.py      + GrokOneTransformer, MoE, RoPE, RMSNorm
│   │   ├── training.py         + architecture flag, MoE metrics logging
│   │   ├── metrics.py          + routing entropy, specialization, collapse
│   │   ├── data.py             + format_for_grok1(), eval suite generator
│   │   └── __init__.py         + bridge exports
│   ├── setup.py
│   └── README.md               → original research documentation
│
└── grok-1-main/                fork of xai-org/grok-1
    ├── model.py                + to_grokking_config(), architecture_summary()
    ├── run.py                  + --eval-grokking mode
    └── README.md               → original model documentation
```

All original files work exactly as before. Bridge code is appended, never replacing. Every addition is marked with a `# Bridge:` comment. Running `git diff` against the upstream repos shows only additions.

---

## Why

Because both projects are named `grok`. Because naming things is hard and irony compounds. Because Musk accused OpenAI of abandoning open source, named his AI after a word meaning deep understanding, then open-sourced it eleven days after filing a lawsuit — while OpenAI had a repository studying how models *learn* to understand, sitting quietly with the same name.

The act of forcing incompatible things to work together has a name in Chinese: 强兼 *(qiáng jiān)*. It seemed appropriate.

The answer to "Does Grok grok grokking?" is, genuinely, not yet known.

---

## License

- `grok-main/` — [MIT License](grok-main/LICENSE)
- `grok-1-main/` — [Apache 2.0](grok-1-main/LICENSE.txt)
- Bridge code (`GrokOneTransformer`, `does_grok_grok.py`, and all `# Bridge:` additions) — public domain

---

<div align="center">
<sub>
<em>"The word is much wider in meaning than any English word conceived to date — it means to understand so thoroughly that the observer becomes a part of the observed."</em><br>
— Heinlein, via Jubal Harshaw
</sub>
</div>
