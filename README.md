<p align="center">
  <h1 align="center">🍼 Moms Verdict Generator</h1>
  <p align="center">
    <strong>Review → Structured Verdict in EN + AR</strong>
  </p>
  <p align="center">
    <em>Turn ~200 messy customer reviews into a grounded, multilingual, schema-validated product verdict — with honest uncertainty handling.</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Pydantic-v2-e92063?logo=pydantic&logoColor=white" alt="Pydantic v2">
  <img src="https://img.shields.io/badge/LLM-OpenRouter-7c3aed?logo=openai&logoColor=white" alt="OpenRouter">
  <img src="https://img.shields.io/badge/Embeddings-sentence--transformers-orange" alt="sentence-transformers">
  <img src="https://img.shields.io/badge/Search-FAISS-yellow" alt="FAISS">
  <img src="https://img.shields.io/badge/tests-134%20passing-brightgreen" alt="Tests">
</p>

---

## 🧩 Problem

### Problem Statement

Given up to 200 customer reviews for a single baby or mom product, synthesize them into a structured **Moms Verdict** in both English and Arabic. The output should summarize the overall customer experience, list grounded pros and cons, provide a clear recommendation, and honestly express uncertainty when the reviews are too sparse, noisy, or contradictory.

Customer reviews are messy — duplicates, spam, noise, mixed languages, conflicting opinions. Extracting a trustworthy product verdict from this chaos requires:

- **Grounding** — every claim must trace back to actual reviews
- **Multilingual output** — native English AND Arabic (not translation)
- **Schema validation** — strict JSON output, always
- **Honest uncertainty** — say "not enough data" when it's true

This prototype solves all four.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Review Input (JSON/TXT)                   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: PREPROCESSOR                                       │
│  • Strip whitespace & normalize Unicode (NFC)               │
│  • Remove exact duplicates (set-based)                      │
│  • Remove near-duplicates (cosine similarity ≥ 0.9)         │
│  • Discard reviews < 3 words                                │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: RETRIEVER (if > 50 reviews)                        │
│  • Embed with sentence-transformers (all-MiniLM-L6-v2)      │
│  • Build FAISS IndexFlatL2                                   │
│  • Select ≤50 via MMR diversity sampling                     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: FEATURE EXTRACTOR (LLM call)                       │
│  • Sentiment distribution (positive/negative/neutral)       │
│  • Themes with source attributions                          │
│  • Complaints & praises with source attributions            │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: AGGREGATOR                                         │
│  • Count theme frequencies                                  │
│  • Rank pros/cons descending by frequency                   │
│  • Filter themes < 2 mentions                               │
│  • Flag low-confidence if < 2 distinct themes               │
└──────────────┬──────────────────────────────┬───────────────┘
               ▼                              ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│  Step 5a: GENERATOR EN   │  │  Step 5b: GENERATOR AR       │
│  Independent LLM call    │  │  Independent LLM call        │
│  → summary_en            │  │  → summary_ar                │
│  → verdict_en            │  │  → verdict_ar                │
│  → pros, cons            │  │  (native Arabic, NOT         │
│                          │  │   translation)               │
└──────────┬───────────────┘  └──────────────┬───────────────┘
           └──────────┬──────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 6: VALIDATOR                                          │
│  • Pydantic v2 schema validation                            │
│  • Retry up to 2x on failure                                │
│  • Deterministic confidence scoring                         │
│  • Fail loudly if all retries exhausted                     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   ✅ Moms Verdict JSON                       │
└─────────────────────────────────────────────────────────────┘
```


---

## ⚡ Quick Start (< 5 minutes)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get your free API key

1. Go to [openrouter.ai](https://openrouter.ai) → sign up (free, no credit card)
2. Create an API key from your dashboard
3. Add it to the `.env` file in the project root:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 3. Start the server

```bash
# Windows
python -m uvicorn app.main:app --reload

# Linux/Mac
uvicorn app.main:app --reload
```

### 4. Open the interactive docs

Visit **http://127.0.0.1:8000/docs** — upload a file and test instantly via Swagger UI.

### 5. Or use curl

```bash
curl -X POST http://127.0.0.1:8000/verdict \
  -F "file=@data/mixed_reviews.json;type=application/json"
```

### Demo Video

The 3-minute walkthrough is included at the repo root as `Shubham_video.mp4`. It shows five end-to-end inputs, including low-data/noisy cases where the system returns low confidence or refuses with a structured error.

---

## 📋 Output Schema (Strict)

Every response is validated against this exact schema:

```json
{
  "summary_en": "Natural English summary grounded in reviews",
  "summary_ar": "ملخص طبيعي بالعربية مبني على المراجعات",
  "pros": ["Positive aspect 1", "Positive aspect 2"],
  "cons": ["Negative aspect 1", "Negative aspect 2"],
  "verdict_en": "Concise English verdict",
  "verdict_ar": "حكم موجز بالعربية",
  "confidence": 0.78,
  "uncertainty_reason": null
}
```

| Field | Type | Rules |
|-------|------|-------|
| `summary_en` | string | Non-empty, grounded in reviews |
| `summary_ar` | string | Non-empty, native Arabic (not translated) |
| `pros` | string[] | Only from reviews. Empty `[]` if none found |
| `cons` | string[] | Only from reviews. Empty `[]` if none found |
| `verdict_en` | string | Concise English verdict |
| `verdict_ar` | string | Concise Arabic verdict |
| `confidence` | float | 0.0–1.0. Below 0.5 = insufficient data |
| `uncertainty_reason` | string \| null | Explains low confidence. `null` when ≥ 0.5 |

---

## 📥 Input Format

Accepts JSON or plain text files with 1–200 reviews:

**JSON** (array of strings):
```json
[
  "Great product, very useful for my baby",
  "Terrible quality, broke in 2 days",
  "Love the design but overpriced"
]
```

**Text** (one review per line):
```
Great product, very useful for my baby
Terrible quality, broke in 2 days
Love the design but overpriced
```

---

## 🔬 Example: Mixed Reviews

### Input
```bash
curl -X POST http://localhost:8000/verdict \
  -F "file=@data/mixed_reviews.json;type=application/json"
```

### Output
```json
{
  "summary_en": "Parents generally appreciate the safety features and comfort of this car seat, though some note concerns about the installation process and price point.",
  "summary_ar": "يقدّر الآباء بشكل عام ميزات الأمان والراحة في مقعد السيارة هذا، رغم أن البعض يشير إلى صعوبات في التركيب وارتفاع السعر.",
  "verdict_en": "A solid car seat with strong safety credentials, but be prepared for a tricky install.",
  "verdict_ar": "مقعد سيارة متين بمعايير أمان قوية، لكن كن مستعدًا لتركيب صعب بعض الشيء.",
  "pros": [
    "Excellent safety ratings and side-impact protection",
    "Comfortable padding for long rides",
    "Durable build quality"
  ],
  "cons": [
    "Installation can be difficult without help",
    "Higher price compared to alternatives"
  ],
  "confidence": 0.78,
  "uncertainty_reason": null
}
```

### Example: Not Enough Data (Failure Case)
```bash
curl -X POST http://localhost:8000/verdict \
  -F "file=@data/few_reviews.json;type=application/json"
```
```json
{
  "summary_en": "Limited reviews suggest a basic baby swing with average performance.",
  "summary_ar": "تشير المراجعات المحدودة إلى أرجوحة أطفال بسيطة بأداء متوسط.",
  "pros": ["Gentle rocking motion"],
  "cons": ["Motor is a bit loud"],
  "verdict_en": "Insufficient data to form a strong verdict.",
  "verdict_ar": "بيانات غير كافية لتكوين حكم قوي.",
  "confidence": 0.15,
  "uncertainty_reason": "Only 3 reviews available; Insufficient theme diversity"
}
```

---

## 🔌 API Reference

### `POST /verdict`

Upload a JSON or TXT file of customer reviews.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | `UploadFile` | JSON array or line-delimited text file |

**Response Codes:**

| Status | Meaning |
|--------|---------|
| `200` | Valid `MomsVerdict` JSON returned |
| `422` | Invalid input, empty file, or all reviews filtered as noise |
| `500` | LLM generation failure or schema validation failure after retries |

**Interactive docs:** http://127.0.0.1:8000/docs

---

## 🧪 Evaluation System

### Run the evaluation suite

```bash
python -m evals.eval
```

### 10 Test Cases

| # | Test Case | Scenario | Expected |
|---|-----------|----------|----------|
| 1 | `all_positive_reviews` | 240 positive stroller reviews | High confidence, pros only |
| 2 | `all_negative_reviews` | 240 negative monitor reviews | High confidence, cons only |
| 3 | `mixed_reviews` | 240 mixed car seat reviews | Both pros and cons |
| 4 | `noisy_spam_reviews` | 240 swing reviews with some noise | Preprocessing filters junk |
| 5 | `few_reviews` | Only 3 reviews | confidence < 0.5 |
| 6 | `conflicting_reviews` | 240 contradictory high chair opinions | Mixed pros/cons |
| 7 | `duplicate_heavy` | 240 diaper bag reviews with repeated themes | Dedup + enough unique signal |
| 8 | `single_review` | Just 1 review | confidence < 0.5 |
| 9 | `multilingual_reviews` | 240 English + Arabic crib reviews | Both languages handled |
| 10 | `empty_after_preprocessing` | All noise/too short | `InsufficientDataError` |

### Metrics Checked Per Test Case

- ✅ **Schema validity** — output conforms to MomsVerdict schema
- ✅ **Hallucination check** — pros/cons keywords exist in input reviews
- ✅ **Coverage** — dominant themes from input appear in output
- ✅ **Confidence calibration** — score matches expected range
- ✅ **Uncertainty consistency** — reason populated iff confidence < 0.5

### Eval Scores

Rubric: each case is scored out of 5 points: schema validity (1), grounded pros/cons (1), theme coverage (1), confidence calibration (1), and correct uncertainty/refusal behavior (1).

| # | Test Case | Score | Notes |
|---|-----------|-------|-------|
| 1 | `all_positive_reviews` | 5/5 | Strong pros, no unsupported cons |
| 2 | `all_negative_reviews` | 5/5 | Strong cons, no invented positives |
| 3 | `mixed_reviews` | 5/5 | Captures both praise and complaints |
| 4 | `noisy_spam_reviews` | 4/5 | Filters obvious junk; occasional weak theme wording |
| 5 | `few_reviews` | 5/5 | Correctly lowers confidence and explains sparse evidence |
| 6 | `conflicting_reviews` | 4/5 | Handles disagreement, but confidence can still be a bit optimistic |
| 7 | `duplicate_heavy` | 5/5 | Dedup keeps repeated reviews from dominating |
| 8 | `single_review` | 5/5 | Correctly expresses uncertainty |
| 9 | `multilingual_reviews` | 4/5 | English/Arabic handled; Arabic wording quality depends on model |
| 10 | `empty_after_preprocessing` | 5/5 | Correct structured refusal via `InsufficientDataError` |

Overall score: **47/50**. Biggest known failures are softer Arabic phrasing on small models and slightly overconfident outputs when reviews are truly contradictory.

---

## ?? Sample Data (12 Files)

| File | Reviews | Purpose |
|------|---------|---------|
| `sample_reviews.json` | 240 | General mixed baby bottle reviews |
| `all_positive.json` | 240 | Uniformly positive stroller reviews |
| `all_negative.json` | 240 | Uniformly negative monitor reviews |
| `mixed_reviews.json` | 240 | Balanced positive/negative car seat reviews |
| `conflicting_reviews.json` | 240 | Contradictory high chair opinions |
| `duplicate_heavy.json` | 240 | Diaper bag reviews with repeated themes and enough unique signal |
| `few_reviews.json` | 3 | Low-confidence baby swing sample |
| `single_review.json` | 1 | Low-confidence diaper bag sample |
| `noisy_spam.json` | 240 | Baby swing reviews with some spam/noise mixed in |
| `multilingual_reviews.json` | 240 | English + Arabic crib reviews |
| `empty_after_preprocessing.json` | 14 | All noise -> `InsufficientDataError` |
| `dummy_reviews.json` | 240 | Portable bottle warmer reviews |

---

## 🎯 Prompt Engineering

Three carefully designed prompts ensure quality output:

| Prompt | Purpose | Key Instructions |
|--------|---------|-----------------|
| `SYSTEM_PROMPT_EN` | English generation | Ground in reviews, forbid hallucination, report uncertainty |
| `SYSTEM_PROMPT_AR` | Arabic generation | Native Arabic phrasing, forbid literal translation, ground in reviews |
| `FEATURE_EXTRACTION_PROMPT` | Feature extraction | Source attribution required, structured JSON output |

**Anti-hallucination safeguards:**
- Every pro/con must be traceable to input reviews
- LLM explicitly instructed: "Do NOT invent information not present in the reviews"
- Arabic prompt written entirely in Arabic to ensure native phrasing
- Uncertainty encouraged when data is sparse or conflicting

---

## ⚖️ Tradeoffs & Design Decisions

| Decision | Benefit | Cost |
|----------|---------|------|
| Free-tier LLM (OpenRouter) | Zero cost | Lower output quality vs GPT-4 |
| Separate EN/AR generation | Native quality in both languages | ~2x latency (two LLM calls) |
| MMR diversity sampling | Better topic coverage | Slower than random sampling |
| Retry-with-validation (2x) | Higher reliability | Added latency on bad generations |
| Deterministic confidence | Transparent, reproducible | Less flexible than LLM self-assessment |
| Near-dedup via embeddings | Catches paraphrased duplicates | Requires model loading overhead |

I picked this problem because customer reviews are a good stress test for grounded generation: there is enough messy language to need an LLM, but enough source evidence to judge whether the output is honest. I rejected a generic chatbot and a pure sentiment classifier because both would hide the hard parts: attribution, multilingual output, and uncertainty.

The model/architecture choice is intentionally practical: FastAPI for a small runnable service, Pydantic for strict output contracts, OpenRouter-compatible chat completion calls for generation, and sentence-transformers + FAISS only when the review set is too large to fit directly. Uncertainty is handled with deterministic confidence signals from review count, theme diversity, contradictions, and preprocessing loss, rather than asking the model to grade itself.

What I cut: persistent storage, auth, a frontend, streaming responses, human review UI, and expensive reranking. What I would build next: richer multilingual evals, source-span citations in the response, a small dashboard for comparing outputs, and model/provider fallback when the free tier is unavailable.

---

## ⚠️ Known Failure Cases

| Scenario | Behavior | HTTP Status |
|----------|----------|-------------|
| Very few reviews (< 5) | `confidence < 0.5`, `uncertainty_reason` populated | 200 |
| All noise/spam reviews | `InsufficientDataError` — all reviews discarded | 422 |
| LLM rate limits | `GenerationError` after retries exhausted | 500 |
| Conflicting reviews | Lower confidence, mixed pros/cons | 200 |
| Invalid file format | `ParseError` with descriptive message | 422 |
| Empty file | `EmptyInputError` | 422 |

---

## 🧰 Tooling Used

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Runtime |
| FastAPI | 0.115.6 | REST API framework |
| Pydantic | v2 | Schema validation |
| OpenRouter API | Free tier | LLM generation (Llama 3.1 8B) |
| sentence-transformers | 3.3.1 | Review embeddings (`all-MiniLM-L6-v2`) |
| FAISS | faiss-cpu | In-memory vector similarity search |
| OpenAI SDK | 1.58.1 | OpenRouter client (compatible API) |
| python-dotenv | 1.0.1 | `.env` file loading |
| pytest | 8.3.4 | Test runner |
| Hypothesis | 6.119.3 | Property-based testing |
| httpx | 0.28.1 | Async HTTP client for testing |
| Codex / ChatGPT | GPT coding assistant | Code review, README tightening, git/LFS setup help |

---

## 🧪 Running Tests

```bash
# Run all 134 unit tests
python -m pytest tests/ -v

# Run tests in quiet mode
python -m pytest tests/ -q

# Run the evaluation suite (requires API key)
python -m evals.eval
```

---

## 📂 Project Structure

```
moms-verdict-generator/
├── app/
│   ├── __init__.py
│   ├── aggregator.py          # Theme frequency counting, pros/cons ranking
│   ├── errors.py              # Custom pipeline error classes
│   ├── feature_extractor.py   # LLM-based sentiment/theme extraction
│   ├── generator.py           # EN/AR generation + confidence scoring
│   ├── main.py                # FastAPI app and POST /verdict endpoint
│   ├── parsers.py             # JSON and text input parsing
│   ├── pipeline.py            # Pipeline orchestrator (wires all stages)
│   ├── preprocessor.py        # Dedup, normalize, noise filtering
│   ├── prompts.py             # LLM prompt templates (EN, AR, extraction)
│   ├── retriever.py           # Embeddings + FAISS + MMR diversity selection
│   └── schema.py              # MomsVerdict Pydantic model + validation
├── data/                      # 12 sample review files
│   ├── sample_reviews.json
│   ├── all_positive.json
│   ├── all_negative.json
│   ├── mixed_reviews.json
│   ├── conflicting_reviews.json
│   ├── duplicate_heavy.json
│   ├── few_reviews.json
│   ├── single_review.json
│   ├── noisy_spam.json
│   ├── multilingual_reviews.json
│   ├── empty_after_preprocessing.json
│   └── dummy_reviews.json
├── evals/
│   ├── __init__.py
│   ├── eval.py                # Evaluation runner (10 test cases)
│   └── test_cases.json        # Test case definitions
├── tests/                     # 134 unit tests
│   ├── __init__.py
│   ├── test_aggregator.py
│   ├── test_feature_extractor.py
│   ├── test_generator.py
│   ├── test_parsers.py
│   ├── test_pipeline.py
│   ├── test_preprocessor.py
│   ├── test_prompts.py
│   └── test_schema.py
├── .env                       # API key (not committed)
├── .gitignore
├── requirements.txt           # Pinned Python dependencies
└── README.md
```

---

## 📜 Strict Constraints

This system enforces:

- ❌ **NO hallucination** — every claim traces to input reviews
- ❌ **NO translation** — EN and AR generated independently
- ❌ **NO invalid JSON** — Pydantic validation with retry
- ❌ **NO fake confidence** — deterministic scoring from data signals
- ❌ **NO empty strings** — `null` used instead
- ✅ **YES uncertainty** — honest "not enough data" when warranted

---

<p align="center">
  Built with ❤️ for moms who deserve honest product verdicts.
</p>
