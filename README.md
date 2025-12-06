# Narrative Similarity Task

**SemEval 2026 Task 4: Narrative Story Similarity and Narrative Representation Learning**

## What This Task Is About

This competition asks participants to build systems that can understand how similar two stories are, not just in terms of the words they use, but in their deeper narrative structure. We're talking about things like theme, plot progression, character arcs, and how the story resolves.

There are two tracks:

* **Track A**: You're given three stories - an anchor and two choices. Your job is to figure out which of the two choices is more similar to the anchor in terms of narrative.
* **Track B**: Create embeddings for stories where the distance between embeddings reflects how narratively similar they are.

We focused on **Track A** for this work.

---

## What We Built

We tried out a bunch of different approaches to solve this problem. Our best model hits **73% accuracy** on the development set, which is about 19% better than the baseline. Here's how all our approaches stacked up:

| Approach | Accuracy | Key Idea |
|----------|----------|----------|
| **Ensemble (Best Model)** | **73.00%** | Combines symbolic reasoning with embeddings, routing between them based on confidence |
| **Hybrid Symbolic-Neural** | **71.00%** | Uses an LLM to extract story elements (themes, events, outcomes) then compares them |
| Fine-tuned Sentence Transformer | 64.00% | Trained a standard embedding model on synthetic story data |
| Zero-shot LLM | 63.50% | Just asked Gemini to compare stories directly |
| Few-shot LLM | 61.50% | Same as above but gave it 3 examples first |
| Baseline Embeddings | 61.50% | Off-the-shelf sentence embeddings with no training |
| Event-Chain Model | 57.00% | Extracted event sequences and aligned them optimally |
| RoBERTa Pairwise Ranking | 62.50% | Fine-tuned a large transformer to rank story pairs |
| Chain-of-Thought | 40.00% | Step-by-step reasoning (didn't work well) |

---

## The Approaches We Tried

### 1. Baseline: Off-the-Shelf Embeddings

**Code:** [Code/baseline_similarity.ipynb](Code/baseline_similarity.ipynb)

We started by testing whether standard sentence embedding models could handle this task. These models convert text into vectors, and you compare vectors using cosine similarity to see how similar two pieces of text are.

We tested three models:
- `all-MiniLM-L6-v2`: Got 55% accuracy
- `all-mpnet-base-v2`: Got 61.5% accuracy (our baseline)
- Longformer: Got 46.5% accuracy

**What we learned:** These models are pretty good at finding stories that use similar words, but they miss the deeper narrative structure. A story about redemption could use totally different words than another redemption story, and these models wouldn't catch that.

---

### 2. Asking an LLM Directly

**Code:** [Code/few_shot_prompting.ipynb](Code/few_shot_prompting.ipynb)

We tried just asking Gemini to compare stories for us. We tested two versions:
- **Zero-shot:** Just ask the model with no examples → 63.5% accuracy
- **Few-shot:** Give it 3 examples first, then ask → 61.5% accuracy

Interestingly, giving examples actually made it slightly worse. We think this is because narrative similarity is complex enough that a few examples don't really help guide the model.

**What we learned:** LLMs understand narrative better than embeddings, but you can't just throw the problem at them and expect great results.

---

### 3. Training Our Own Embedding Model

**Code:** [Code/fine_tuning.ipynb](Code/fine_tuning.ipynb)

We took the baseline embedding model and fine-tuned it on about 6,773 synthetic story examples that we generated. We used triplet loss, which teaches the model that similar stories should have close embeddings and dissimilar ones should be far apart.

Training details:
- Started with `all-mpnet-base-v2`
- Used a very low learning rate (5e-6) to avoid messing up what it already knows
- Trained for just 1 epoch with batches of 8 examples

**Results:** 64% accuracy (about 2.5 points better than baseline)

**What we learned:** Fine-tuning helps, but not by much. The task needs more than just better embeddings.

**Model saved at:** [finetuned_narrative_model/](finetuned_narrative_model/)

---

### 4. Hybrid Symbolic-Neural (Big Jump)

**Code:** [Code/hybrid_symbolic_neural.ipynb](Code/hybrid_symbolic_neural.ipynb)

This is where things got interesting. Instead of comparing stories directly, we broke down each story into its narrative components first, then compared those.

**How it works:**
1. **Extract story elements** using Gemini:
   - What are the main themes?
   - What key events happen?
   - How does it end?
   - How do characters change?
   - What kind of conflict is there?

2. **Compare the elements** in a structured way:
   - Do the stories share themes?
   - Do similar events happen in both?
   - Do they end similarly?

**Results:** 71% accuracy (9.5 points better than baseline)

**What we learned:** Breaking stories into their narrative building blocks before comparing them is way more effective than comparing the raw text. It's like comparing blueprints instead of looking at two houses and guessing if they were built the same way.

---

### 5. Event-Chain Model

**Code:** [Code/event-chain-model-improved.ipynb](Code/event-chain-model-improved.ipynb)

We tried extracting the sequence of events from each story and then optimally aligning those sequences to see how similar they are.

**How it works:**
- Extract events (who did what to whom)
- Use the Hungarian algorithm to match events between stories
- Calculate how well event sequences align
- Combine with theme and outcome similarity

**Results:** 57% accuracy on full dev set, though with heavy tuning got 65% on a smaller set

**What we learned:** Event alignment is a solid idea, but it's really sensitive to how well you extract events and how you weight different components. Small changes in the weights completely change performance.

---

### 6. Fine-tuned RoBERTa

**Code:** [Code/roberta_pairwise_ranking.ipynb](Code/roberta_pairwise_ranking.ipynb)

We fine-tuned a large RoBERTa model to directly rank story pairs. Each training example is a story pair with a label saying whether they're similar or not.

**How it works:**
- Take each triplet and turn it into two pairs: (anchor, choice A) and (anchor, choice B)
- Train RoBERTa to give higher scores to similar pairs
- Pick whichever choice gets the higher score

**Results:** 62.5% accuracy (best performance was after just 1 epoch, then it started overfitting)

**What we learned:** Even a big model like RoBERTa struggles with this task when fine-tuned directly. The margins between scores were tiny, suggesting it wasn't learning strong distinctions.

---

### 7. Chain-of-Thought Reasoning

**Code:** [Code/chain_of_thought_reasoning.ipynb](Code/chain_of_thought_reasoning.ipynb)

We tried getting the LLM to show its work by scoring different aspects of narrative similarity step-by-step.

**How it works:**
- Extract story elements
- Score theme overlap (0-10)
- Score event similarity (0-10)
- Score outcome similarity (0-10)
- Score conflict type match (0-10)
- Pick whichever choice has the higher total score

**Results:** 40% accuracy (worse than random guessing)

**What we learned:** The model developed a strong bias toward choosing option B. When we looked at the scores, it was giving wildly different scores to very similar stories. This approach didn't work and we dropped it.

---

### 8. Ensemble - Our Best Model

**Code:** [Code/ensemble_approach.ipynb](Code/ensemble_approach.ipynb)

Our best result came from combining the hybrid symbolic-neural approach with the fine-tuned embeddings in a smart way.

**How it works:**
The embedding model calculates how similar each choice is to the anchor. If the difference between those similarities is large (confidence > 0.30), we trust the embedding model's choice. Otherwise, we defer to the symbolic model.

In practice, the symbolic model handles 99% of cases because the embedding model is rarely confident enough.

**Why this works:**
- The symbolic model is generally better (71% vs 64%)
- But sometimes the embedding model catches things the symbolic model misses
- They disagree about 43.5% of the time
- When they disagree, symbolic is right about 60% of the time

**Results:** 73% accuracy (2 points better than symbolic alone)

We tried other ensemble methods too:
- Simple voting: 72.5%
- Score averaging: 72.5%
- Logistic regression: 62.5%
- Gradient boosting: 65%

**What we learned:** Combining models helps, but only if you do it carefully. Simple averaging isn't as good as smart routing.

---

### 9. Generating Training Data

**Code:** [Code/synthetic_data_generation.ipynb](Code/synthetic_data_generation.ipynb)

Since we didn't have much labeled data, we generated our own using Gemini.

**How it works:**
- Give Gemini a few examples of story triplets
- Ask it to generate new ones with diverse themes, time periods, and genres
- Keep the stories to 150-300 words
- Make them sound like Wikipedia summaries

**Results:** Generated about 1,000 examples successfully

We combined these with another 1,897 examples generated by other LLMs (GPT-4, Claude, etc.) for a total of about 6,773 training examples.

**What we learned:** LLM-generated training data is useful for fine-tuning, though you need to make sure it's diverse enough to be helpful.

---

## What We Learned Overall

Here are the big takeaways from all this experimentation:

**1. Narrative similarity is not word similarity**
Standard embedding models max out around 61% because they look for similar words, not similar story structures. Two stories can tell the same narrative using completely different words.

**2. Structure matters more than you'd think**
Our best approaches all broke stories down into components (themes, events, outcomes) before comparing them. This structure-first approach beat everything else.

**3. LLMs understand narrative, but you need to guide them**
Just asking an LLM to compare stories gets 63.5%. Extracting structure first and then comparing gets 71%. The difference is in how you use the LLM.

**4. Combining models helps, but not dramatically**
Our ensemble only added 2 percentage points over the symbolic approach alone. Most of the gains come from picking the right approach, not from combining many approaches.

**5. Some ideas don't work**
Chain-of-thought reasoning flopped (40%), and large model fine-tuning barely beat baselines (62.5%). Not every sophisticated approach pays off.

---

## About the Competition

### The Two Tracks

**Track A** (what we worked on):
You get three stories - one anchor and two choices. Your system needs to decide which choice is more narratively similar to the anchor. The data looks like this:

```json
{
  "anchor_text": "A story about finding love.",
  "text_a": "Another story about finding love.",
  "text_b": "Unrelated text.",
  "text_a_is_closer": true
}
```

The official baseline for this track uses GPT-4o-mini with basic prompting.

**Track B**:
Build an embedding model where the distance between story embeddings matches how similar humans think the stories are. Your embeddings need to be between 10 and 8192 dimensions, and they have to be generated independently for each story (no looking at pairs). The official baseline is SentenceBERT's `all-MiniLM-L6-v2`.

### The Data

The competition provides about 1,000+ annotated story triplets, all sourced from Wikipedia:
- **Sample set**: 39 examples with labels
- **Dev set**: 200 examples with labels (this is what we used for testing)
- **Test set**: 400 triplets + 849 individual stories (labels released after evaluation)
- **Official synthetic set**: 1,900 LLM-generated triplets for training

We also created our own synthetic training data - about 6,773 examples generated using various LLMs.

All data is licensed under CC-BY-SA-4.0.

### What Makes This Task Hard

Narrative similarity isn't about using the same words or phrases. Two stories can be narratively similar while using completely different vocabulary. For example:
- A story about a fisherman finding peace at sea
- A story about a programmer finding peace through meditation

These might share the same narrative arc (journey → struggle → inner peace) but have totally different settings, characters, and word choices. Standard text similarity models fail at this because they look for lexical overlap.

### Why It Matters

Better narrative similarity systems could help with:
- Finding stories with similar themes in large databases
- Recommending books or movies based on narrative structure
- Helping writers understand story patterns
- Analyzing how stories evolve across cultures or time periods

---

## More Information

Check out the official competition page for full details and updates:
[https://narrative-similarity-task.github.io/](https://narrative-similarity-task.github.io/)

If you use this work, cite:
```
SemEval 2026 Task 4: Narrative Story Similarity and Narrative Representation Learning
https://narrative-similarity-task.github.io/
```
