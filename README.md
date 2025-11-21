# Narrative Similarity Task

**SemEval 2026 Task 4: Narrative Story Similarity and Narrative Representation Learning**

## Overview

The Narrative Similarity Task invites participants to build systems that understand and model *narrative similarity*. The goal is to measure how similar two stories are in terms of theme, course of action, outcomes, etc.

There are **two tracks**:

* **Track A**: Given a triple (anchor story, choice A, choice B), decide which choice is **more narratively similar** to the anchor.
* **Track B**: Produce vector embeddings for stories such that cosine similarity correlates with narrative similarity judgments.

## Task Details

### Track A

* Input: JSONL objects containing:

  * `anchor_text`
  * `text_a`
  * `text_b`
  * `text_a_is_closer`

Example:

```json
{
  "anchor_text": "A story about finding love.",
  "text_a": "Another story about finding love.",
  "text_b": "Unrelated text.",
  "text_a_is_closer": true
}
```

* Output: A decision on whether `text_a` is closer to the anchor than `text_b`.
* Baseline: A GPT‑4o‑mini prompting method.

### Track B

* Input: Individual story texts.
* Output: Embeddings (10–8192 dimensions) generated *only* from the individual story, not from cross‑story interactions.
* Goal: Cosine similarity between embeddings should match human similarity judgments.
* Baseline: SentenceBERT `all‑MiniLM‑L6‑v2`.

## Data

### Composition

* **1,000+ annotated triples** (stories from Wikipedia)
* **Sample set**: 39 items (with labels)
* **Development set**: 200 items (with labels)
* **Synthetic training set**: 1,900 LLM‑generated triples
* **Test set**: 400 triples + 849 individual stories (labels released after evaluation)

### Formats

**Track A format:**

```json
{
  "anchor_text": "...",
  "text_a": "...",
  "text_b": "...",
  "text_a_is_closer": true
}
```

**Track B format:**

```json
{
  "text": "This is the story."
}
```

**Synthetic Data Formats**

* Classification format includes a `model_name` field
* Contrastive format includes:

  * `anchor_story`
  * `similar_story`
  * `dissimilar_story`
  * `model_name`

**License:** CC‑BY‑SA‑4.0

## Submission & Evaluation

Your submission must be a ZIP containing:

```
track_a.jsonl
track_b.{npy|jsonl}
```

The leaderboard requires declaring:

1. Which track(s) you're entering
2. The type of method used

### Rules

* Track B embeddings must be produced **independently per story** (no pairwise inference).
* Embedding length must be **10–8192 dimensions**.

## Why This Task Matters

Narrative similarity is a key dimension of understanding stories beyond surface lexical similarity. Systems developed for this task can improve story retrieval, thematic clustering, summarization, creative writing tools, and narrative analytics.

## Getting Started

1. Download the sample, dev, and synthetic data.
2. Review baseline models for both tracks.
3. Choose your track(s).
4. For Track A: Build a model or prompt to classify narrative similarity.
5. For Track B: Build or fine‑tune an embedding model.
6. Export predictions/embeddings in the required format.
7. Submit via CodaBench.

## More Information

Visit the official site for updates and full guidelines:
[https://narrative-similarity-task.github.io/](https://narrative-similarity-task.github.io/)
