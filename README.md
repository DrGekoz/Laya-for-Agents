<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-lockup-dark.png" />
    <img src="assets/logo-lockup.png" alt="Laya" width="300" />
  </picture>
</p>

<h1 align="center">Laya for Agents</h1>

<p align="center"><strong>Local typed decisions for agent harnesses.</strong><br />
A <a href="https://github.com/NandhaKishorM/laya">Laya</a> decision engine served over the
TypeSafe Jev <code>/v1/systemone</code> wire protocol, so
<a href="https://github.com/kerpopule/hermes-jev-skills">hermes-jev-skills</a> can route models,
filter memory, pick compactions, choose skills and drive computer/browser steps
<em>on your own hardware</em>.</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" /></a>
  <img alt="Python" src="https://img.shields.io/badge/python-%3E%3D3.10-3776ab.svg" />
  <img alt="Protocol" src="https://img.shields.io/badge/protocol-Jev%20%2Fv1%2Fsystemone-6f42c1.svg" />
  <img alt="Runs" src="https://img.shields.io/badge/runs-CPU%20%7C%20CUDA-76b900.svg" />
  <img alt="Tests" src="https://img.shields.io/badge/tests-121%20passing-brightgreen.svg" />
  <img alt="Fork of" src="https://img.shields.io/badge/fork%20of-NandhaKishorM%2Flaya-orange.svg" />
  <img alt="For" src="https://img.shields.io/badge/built%20for-hermes--jev--skills-4b8bbe.svg" />
  <img alt="No API key" src="https://img.shields.io/badge/API%20key-not%20required-success.svg" />
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#the-problem-this-solves">The problem</a> ·
  <a href="#the-wire-protocol">Wire protocol</a> ·
  <a href="#hermes">Hermes</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#limits-worth-measuring">Limits</a> ·
  <a href="#what-this-is-built-on">Credits</a>
</p>

---

## What this is

TypeSafe's [Jev](https://docs.typesafe.ai) has a good idea behind it. An agent burns
frontier-model tokens on work that is not thinking — which model should answer this turn, which
of the retrieved passages are worth reading, which turns survive a summary, which button comes
next. Those are *decisions*, not prose. A small model answers them in a fraction of a second
with a calibrated probability attached, and the expensive model goes back to writing.

[Laya](https://github.com/NandhaKishorM/laya) is a local, non-autoregressive decision engine with
the same three primitives — `choice`, `score` and `noul` (probability of a yes) — answered in a
single forward pass. It never generates text, so there is nothing to parse and nothing to
hallucinate. It ships a server that speaks Jev's wire protocol so a Jev client can point at it.

**This project is that join, made safe.** It is a fork of Laya that adds one module — the bridge
— and the bridge exists for one reason.

## The problem this solves

Laya's `predict` scores a state from a **single window** and silently discards everything past
`max_len`. That defaults to **512 tokens**, and the stock server offers no environment variable to
raise it. In the command-line path `max_len` is a per-request argument, so you are expected to
pass it. Agent clients do not, and the reason is not carelessness: the cloud engine they were
written against reads the whole state, so there was never a budget to pass.

So the failure is not an error. It is a **well-formed answer computed from the first half of the
input** — a reply that passes every schema check on the client, carries a confident probability,
and gets acted on.

Here is the arithmetic for the four things an agent actually asks, against that 512-token window:

| what the agent asks | what it sends | tokens | at `max_len=512` |
|---|---|---|---|
| model routing | the turn, redacted, up to 2,500 chars | ~625 | **cut** |
| skill selection | the turn + a batch of skill names | varies | **cut** |
| choosing turns | ~71 turns, first/last 350 chars each | ~12,000 | **cut** |
| memory ranking | up to 60 passages, 900 chars each | up to ~15,000 | **cut** |

Every row is a decision made from a fragment, reported with full confidence.

Laya for Agents makes that impossible:

* **`single`** — the state fits one comfortable pass. `max_len` is raised to cover it exactly, and
  the model answers in one forward pass.
* **`scan`** — the state is too large. It is scored window by window with `Router.predict_long`
  and aggregated per question (`noul` takes the strongest window; `choice` and `score` the most
  confident one). Slower, and **nothing is discarded**.

Both modes are reported in every reply, so a caller can tell a single-pass decision from a scanned
one without reading a log.

## The second problem: the same word, a different number

The truncation above is the loud failure. This is the quiet one, and it was found by running a
real client against a real checkpoint and watching nothing happen.

An agent client gates on `confidence`. Laya and Jev report a **different quantity under that same
name**:

| | formula | a 4-option answer at p_max = 0.7 scores |
|---|---|---|
| **Jev** | `(n·p_max − 1) / (n − 1)` | **0.60** |
| **Laya** (on `choice`/`score`) | `1 − H(p)/log(k)` — normalized entropy | **0.25** |

Same answer, same probabilities, genuinely well calibrated — and a client with a 0.6 threshold
reads one as "confident" and the other as "no opinion". Every decision then falls through to the
fail-open path, so the feature is not broken: it is **silent**, which is worse, because the log
looks clean and the model never changes.

Measured on the live engine before this was fixed, `route.decide()` on four coding prompts:

```
BEFORE, Laya's entropy value served as `confidence`:

trivial lookup    tier=None  "low confidence 0.25"   -> kept the current model
ordinary work     tier=None  "low confidence 0.24"   -> kept the current model
hard coding       tier=None  "low confidence 0.22"   -> kept the current model
expert + risky    tier=medium general                 (the one that cleared)

AFTER, Jev's formula served instead:

a small lookup    conf 0.36   a coding turn   conf 0.40
```

The metric is now the one the client's thresholds were written for. It is worth being precise about
what is and is not fixed: a genuine coding turn really does sit around 0.36–0.40 on a 4-level
rubric, so jevkit's stock 0.6 threshold still will not move the model on an ordinary turn. That
second part is legitimate configuration, not a mismatch — which is why the step below is still the
one that matters. Start in shadow mode, read the log, and set your own numbers.

Laya's own documentation says it plainly — *"never compare the two against one threshold"* — and
this is what that means in practice.

So a reply carries the metric the client was written against, computed from the probabilities Laya
actually produced. The engine's own value is never lost: it is kept beside it as
`confidence_laya`, and `LFA_CONFIDENCE_STYLE=laya` passes it through untouched for anyone who
would rather gate on entropy directly.

```json
"queue": {"type": "choice", "choice": "billing",
          "probabilities": {"billing": 0.9281, "tech": 0.0412, "other": 0.0307},
          "confidence": 0.8921,        // Jev's formula -- what the client gates on
          "answer_confidence": 0.9281, // Laya's calibrated max(p), untouched
          "confidence_laya": 0.4534}   // Laya's entropy value, preserved
```

`noul` is left alone under either style: with two outcomes Laya already reports
`max(p_yes, 1−p_yes)`, which is the number a yes/no gate wants.

## Quick start

```bash
git clone https://github.com/DrGekoz/Laya-for-Agents
cd Laya-for-Agents
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[serve]"

laya-for-agents serve                             # http://127.0.0.1:8000
```

Then point any Jev client at it. **No API key is needed** — and none is sent, because the Jev
client deliberately never forwards a provider credential to an override host:

```bash
export TYPESAFE_BASE_URL=http://127.0.0.1:8000
```

Ask it something directly:

```bash
curl -s http://127.0.0.1:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "We were billed twice for March. Refund it today or we cancel.",
  "questions": {
    "queue":   {"type": "choice", "instructions": "Which team?",
                "criteria": {"billing": "billing and refunds", "tech": "login and app issues",
                             "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["calm", "firm", "angry", "furious"]},
    "churn":   {"type": "noul", "instructions": "Does the customer threaten to leave?"}
  }
}'
```

Then check the install, see it answer a real client, and wire up Hermes:

```bash
laya-for-agents doctor          # torch, device, checkpoints, port, budget, Hermes env
laya-for-agents smoke           # load a checkpoint and answer one real question set
laya-for-agents serve           # then, in another terminal:
python lfa_verify_live.py       # drives hermes-jev-skills' own client at this server
laya-for-agents setup-hermes    # install hermes-jev-skills and point it at this server
```

`lfa_verify_live.py` is the one that proves the wiring: it imports the consumer's
`jevkit.client` and calls `ask()` for real, so the reply has to survive *that project's* strict
validator, and then runs `route.decide()` — the actual model-routing entry point.

## The wire protocol

One route, the same as Jev's and the same as Laya's own server.

### `POST /v1/systemone`

| field | required | meaning |
|---|---|---|
| `state` | yes | text, email, ticket or JSON document to decide on |
| `questions` | yes | object keyed by question id; each is `choice` / `score` / `noul` |
| `model` | no | an explicit checkpoint; an unrecognised id (a Jev id like `jev-latest`) means "let the router choose" |
| `task`, `lang`, `lang_guess` | no | checkpoint selection, language detection overrides |
| `max_len`, `head_max_len` | no | per-request token budget; raised to cover the request when absent |
| `min_confidence` | no | abstention threshold; a below-threshold answer is marked `low_confidence` |

Hook arguments (`hooks`, `on_predict_start`, `on_predict_end`, `hooks_raise`, `hooks_timeout`) are
**refused with a 422 rather than ignored**. A hook is a callable that runs inside this process, so
no value a caller sends can have a meaning here — silently dropping one would report a request as
honoured when it was not.

### Response

```json
{
  "model": "laya-for-agents",
  "answers": {
    "queue": {"type": "choice", "choice": "billing",
              "probabilities": {"billing": 0.9281, "tech": 0.0412, "other": 0.0307},
              "confidence": 0.4534, "answer_confidence": 0.9281},
    "churn": {"type": "noul", "noul": 0.7712, "confidence": 0.7712}
  },
  "usage": {"input_tokens": 18, "output_tokens": 0},
  "routing": {"model": "english", "repo": "convaiinnovations/laya",
              "reason": "English Latin text"},
  "laya": {"mode": "single", "estimated_state_tokens": 18, "single_max": 1024,
           "note": "answered in one pass with the state budget raised to fit"}
}
```

`model`, `answers`, `usage` and `routing` are the keys a Jev client reads. The `laya` block is
additive and is where the behaviour that is *not* Jev's becomes visible: which mode ran, how big
the state was estimated to be, and how many windows were scanned.

### `GET /health`

Always open, never touches the inference gate, and **answers while checkpoints are still loading**.
Reports resident checkpoints, the device, the budget policy in force, and counters — so a deployment
can confirm what it is really serving.

```json
{
  "status": "ok",
  "ready": true,
  "loading": false,
  "loading_for_s": null,
  "loaded": ["english", "multilingual"],
  "device": "cpu",
  "budget": {"single_max": 1024, "long_policy": "multilingual", "scan_max_tokens": 8192,
             "confidence_style": "jev"},
  "stats": {"requests": 12, "single": 12, "multilingual": 0, "scan": 0, "errors": 0, "truncated": 0}
}
```

Gate a **supervisor** on `ready`, not on the status code. `status` stays `ok` throughout, because
the HTTP surface genuinely is serving — a plain "is it up" probe that fails for three minutes gets
the process killed and restarted in a loop. `loading` and `loading_for_s` say a load is in flight;
`loaded` fills in checkpoint by checkpoint, so a working start looks like:

```
   9s  status=ok  loading=True   ready=False  loaded=[]                       loading_for_s=16.9
  76s  status=ok  loading=True   ready=False  loaded=['english']              loading_for_s=83.9
 105s  status=ok  loading=False  ready=True   loaded=['english','multilingual']
```

Getting that right took a real fix, and the reason is worth knowing if you touch this code:
`Router.load` in upstream Laya holds the Router's lock **across the whole `Agent(...)` build**, and
`Router.loaded` takes that same lock. Introspecting through the documented property during a load
therefore blocks for minutes. On a health endpoint that is worse than being down — the connection is
accepted (so a port check says "up") and the reply never arrives (so the client hangs to its
timeout). `describe()` reads the registry without the lock instead, and skips the property entirely
while loading.

### Errors

| status | when |
|---|---|
| `400` | body is not JSON, has no `questions`, or `state` is missing or null |
| `401` | `LFA_API_KEY` is set and the bearer is missing or wrong |
| `413` | a protocol cap: state length, questions, choice options, score levels, or a budget over the ceiling |
| `422` | a question that cannot be encoded, a malformed control, or a refused hook argument |
| `500` | inference failed — always the fixed string `inference failed`, so paths and weights never leak |
| `503` | more requests in flight than `LFA_MAX_CONCURRENT` |

Over-cap load is **refused, not queued**, so a slow client cannot starve `/health`.
Every non-200 is a fail-open on a Jev client, which is exactly the right outcome for a request
this server cannot answer correctly.

## Hermes

[hermes-jev-skills](https://github.com/kerpopule/hermes-jev-skills) is the other half. It ships
eleven `SKILL.md` skills, a Hermes plugin and a `jev` CLI. Its plugin uses only public Hermes
seams — `pre_llm_call`, the `llm_request` middleware, `transform_llm_output`,
`transform_tool_result`, `pre_approval_request` / `post_approval_response`, tools and a slash
command — so `hermes update` does not break it and nothing in Hermes core is patched.

```bash
laya-for-agents setup-hermes
```

That clones the skill pack, runs its installer, and writes `TYPESAFE_BASE_URL` into the Hermes
`.env` (backing the file up first). Then restart the gateway and start in shadow mode:

```
/jev                        status
/jev routing shadow         decide and log, switch nothing  <-- start here
/jev routing on             switch models per turn
/jev skills on              suggest the right skill per turn
/jev notice on              show "[Jev] hard · coding → <model> · confidence 0.92"
```

`jev doctor` should report the override as valid with no key:

```json
{
  "key": {"present": false, "provider": "absent"},
  "endpoint_override": {"url": "http://127.0.0.1:8000/v1/systemone",
                        "valid": true, "bearer": null},
  "jev": {"reachable": true}
}
```

Two things about the Hermes side worth knowing before you turn anything on:

**`jev plan` is not Jev.** The computer-use `--plan` feature (planning a multi-step command once,
up front) calls an OpenAI-compatible *text* model at `TEXT_MODEL_BASE_URL`, defaulting to
OpenRouter. For a fully local setup, point that at whatever you already run locally and set
`JEV_PLAN_MODEL`. Everything else — the per-step action choice — goes through the local engine.

**Re-tune the confidence thresholds.** The routing config gates on `min_confidence`,
`simple_needs_confidence`, `hard_needs_probability` and `simple_needs_probability`, and those
defaults were tuned against Jev's calibration. Laya's `confidence` on `choice`/`score` is
normalized entropy `1 - H(p)/log(k)` where Jev's is `(n·p_max - 1)/(n - 1)` — different
quantities, and Laya's own documentation says never to compare them against one threshold. Start
in shadow mode, read the log, and set your own numbers.

## Configuration

Everything is an environment variable.

### Surface

| variable | default | meaning |
|---|---|---|
| `LFA_HOST` | `127.0.0.1` | bind address |
| `LFA_PORT` | `8000` | bind port |
| `LFA_API_KEY` | none | when set, every decision needs `Authorization: Bearer …` |
| `LFA_MAX_CONCURRENT` | `16` | requests admitted at once; excess gets `503` |
| `LFA_LOG_LEVEL` | `info` | uvicorn log level |

### Model

| variable | default | meaning |
|---|---|---|
| `LFA_DEVICE` | auto | torch device for every checkpoint |
| `LFA_PRELOAD` | `1` | build the checkpoints at startup, not lazily |
| `LFA_MODELS` | all | checkpoints to preload (`english,multilingual,typed-decisions`) |
| `LFA_MAX_LOADED` | `2` | checkpoints kept resident at once |
| `LFA_THREADS` | torch default | cap intra-op threads on CPU; keep it at or below physical cores |

### Budget — the part that matters

| variable | default | meaning |
|---|---|---|
| `LFA_SINGLE_MAX` | `1024` | largest request answered in one pass; past it, the long policy takes over |
| `LFA_LONG_POLICY` | `multilingual` | `multilingual` = one pass on the 8,192-token checkpoint; `scan` = window-by-window |
| `LFA_MULTILINGUAL_BUDGET` | `8192` | the long checkpoint's usable window |
| `LFA_SCAN_MAX_TOKENS` | `8192` | a scan past this is refused rather than run — see the latency note below |
| `LFA_WINDOW_FLOOR` | `512` | never hand the model a tighter window than this |
| `LFA_BUDGET_SLACK` | `64` | headroom over the size estimate |
| `LFA_MAX_TOKEN_BUDGET` | `8192` | ceiling on a caller's own `max_len` |
| `LFA_CONFIDENCE_STYLE` | `jev` | which `confidence` a reply carries; `laya` passes the engine's own through |
| `LFA_AUTO_LONG` | `1` | allow the long paths at all |
| `LFA_LONG_WINDOW` | checkpoint default | tokens per scan window |
| `LFA_LONG_STRIDE` | library default | window overlap |
| `LFA_HEAD_MAX_LEN` | `192` | tokens spent on the question side |
| `LFA_HEAD_MAX_LEN_CEILING` | `512` | how far `head_max_len` may be widened for a wide question |
| `LFA_MARKER_TOKENS` | `4` | tokens Laya spends wrapping each option in the size estimate |
| `LFA_TOKENS_PER_CHAR` | `4.0` | divisor for the byte-based size estimate |

`LFA_SINGLE_MAX` is the main dial. Raise it to trade cost for a single forward pass; `1024` sits
inside every shipped checkpoint's positional range. Lower it if you would rather send more work to
the long checkpoint.

### Latency, measured on CPU (16 cores, no GPU)

| path | measured |
|---|---|
| checkpoint load, warm disk | 92 s (all three) / 105 s (english + multilingual) |
| checkpoint load, cold disk | 218 s |
| one routing-sized turn, first call | 6.0–9.5 s (includes warm-up) |
| one routing-sized turn, warm | **1.27 s** median (min 1.22, max 1.38); 0.98 s observed later |
| a 7,700-token state, window-by-window scan | **182 s** over 42 windows |

That last row is why `LFA_SCAN_MAX_TOKENS` defaults to the multilingual window: a decision that
arrives three minutes late is worth less than one that fails open immediately. Warmed, a single
pass is comfortably inside the 2.5 s budget a Jev client gives routing. On a GPU expect the single
pass in tens of milliseconds and the scan in seconds — the policy is the same, only the clock
changes.

## Limits worth measuring

From Laya's own published limits, plus what the two clients expect. Measure these on your own
traffic before trusting a confidence gate.

* **Keep a choice question to roughly 20 options.** Option descriptions share a fixed token
  budget, and the 77-label Banking77 run performs poorly at the default budget. The bridge widens
  `head_max_len` up to the ceiling to fit a wide question, and returns a `422` naming the question
  when it still does not fit — but the ceiling is a squeeze, not a fix. Shortlist first, then choose.
* **`score` is the weakest primitive**, and the multilingual checkpoint has a measured bias
  against the first score level. Model routing's `difficulty` question is a score question.
* **`noul` can follow its option labels rather than the state.** Relevant to the injection-screen
  questions, where following the wrong thing is the failure that matters.
* **Long documents are reliable to about 4,000 tokens** with the multilingual checkpoint at
  `max_len=8192`; beyond that accuracy is variable. Scanning keeps the whole state, but it does not
  make the encoder better at long inputs.
* **Calibration is not transferable.** Both base checkpoints are over-confident as shipped. Fit
  temperatures on held-out examples from your own workflow before gating on confidence. The
  published checkpoints actually emit this warning at load:

  ```
  RuntimeWarning: laya: this checkpoint ships invalid temperatures or values outside [0.5, 5];
  using choice:11+=0.10058280825614929 -> 0.5. Treat confidence from the affected entries as
  uncalibrated.
  ```

  That is Laya clamping a bad temperature for choices with 11 or more options, and it means
  `confidence` on a wide choice is not calibrated at all. Treat it as a ranking, not a probability.
* **Latency does not transfer.** Published figures are T4 numbers; warm and cold CPU calls are a
  different story (see the table above). The `/health` counters and the `latency_ms` in each reply
  are there for this.

## Layout

```
laya_for_agents/          the bridge — this is what the fork adds
  settings.py             every knob, from the environment
  protocol.py             the Jev wire protocol: validation, estimation, confidence, envelopes
  engine.py               the Router plus the budget policy that makes it safe
  server.py               the HTTP surface (FastAPI/ASGI)
  cli.py                  serve · doctor · smoke · config · setup-hermes
tests/
  test_lfa_protocol.py    121 offline tests: shapes, caps, budgets, dispatch, error mapping
  test_lfa_engine.py
  test_lfa_server.py      the real ASGI app over a real socket, with a fake engine
lfa_verify_live.py        end-to-end against a running server, using the consumer's own client
LayaForAgents.bat         launcher
laya/                      upstream Laya, unchanged
docs/ research/ sdk/ …     upstream documentation, benchmarks and SDKs, unchanged
README-LAYA.md             upstream Laya's own README, kept verbatim
NOTICE                     attribution for all three projects
```

The upstream tests, docs, benchmarks and SDKs are untouched. Sync with the original with:

```bash
git fetch upstream && git merge upstream/main
```

## Tests

```bash
python -m unittest discover -s tests -p "test_lfa_*.py"
```

121 tests, no torch and no checkpoint required — a fake router stands in, and the HTTP tests run
the real ASGI app on a real socket. They pin the things that would otherwise fail silently: that a
long state is *sent to the long checkpoint* rather than cut, that no request is ever given a window
smaller than it needs, that a scan too large to finish in time is refused instead of hanging, that a
wide question widens `head_max_len` instead of being refused, that a Jev model id is dropped rather
than forwarded (forwarding it is a 500 on every request), that the reply carries the confidence
metric the client gates on, and the invariants the Jev client enforces (probabilities cover every
offered option and sum to one, the reported choice is the argmax, a score equals the mean of its own
distribution).

## What this is built on

Two projects, neither of them mine, joined by one module.

**[NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)** — Apache-2.0, by Nandakishor M /
Convai Innovations. The decision engine itself: a multilingual, non-autoregressive System 1 model
that answers typed `choice`, `score` and `noul` questions over any state in a single forward pass,
trained with reinforcement learning against strictly proper scoring rules, with a router that picks
the checkpoint per request. Three checkpoints — `laya` (ModernBERT-large, 421M, 512 context),
`laya-multilingual` (mmBERT-base, 322M, 1024 up to 8192) and `laya-typed-decisions` (421M) — plus
`laya-serve`, whose `/v1/systemone` implementation is what this fork serves. Laya is the model and
the wire protocol; this project is the budget policy on top of it.

**[kerpopule/hermes-jev-skills](https://github.com/kerpopule/hermes-jev-skills)** — MIT, by Steve
Darlow. The consumer this is built for: eleven agent-agnostic `SKILL.md` skills and a Hermes plugin
that hand the small decisions to a Jev-shaped engine — model routing, search, social research,
memory filtering, web screening, handoffs, turn selection, skill selection, triage, mailbox sorting,
computer use and browser use. Its `TYPESAFE_BASE_URL` override and its reply validator are what make
this fork possible and what it is tested against; its `client.ask()` docstring is where the
"well-formed answer to the wrong half of the input" hazard becomes visible.

**[TypeSafe Jev](https://docs.typesafe.ai)** — the original decision model and the protocol both
projects above speak. Laya for Agents implements its wire protocol, not its weights, and is not
affiliated with TypeSafe.

## License

Apache-2.0, inherited from Laya. Upstream copyright and attribution are unchanged; see
[LICENSE](LICENSE) and [NOTICE](NOTICE).
