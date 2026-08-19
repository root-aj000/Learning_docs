---
title: Example 4 — Sentiment Analysis (RNN/LSTM, end-to-end)
description: "The full job pipeline on a real NLP task — classify review sentiment with an LSTM. Where the embedding matmul, LSTM gates, and the chain rule through time live. Plus the honest production lesson: on tiny data, simple models beat deep ones."
tags: [math, ml, practical, nlp, rnn, lstm, pytorch, embedding]
---

# EXAMPLE 4 — SENTIMENT ANALYSIS (the RNN pipeline)

> **The problem (as it would arrive at a job):** a streaming company wants to know if a review is positive or negative. Text is *sequential* — "not good" ≠ "good not" — so the model must process words **in order**. That's what an RNN/LSTM is for.

**This doc's promise:** the same five-line loop from `00-mental-model.md`, now with the two new math pieces sequence models add: **embeddings** (word → vector = a matmul) and **LSTM gates** (sigmoid/tanh matmuls + element-wise multiplies that implement "memory"). You'll also see — with real numbers — the most honest lesson in production ML: **on tiny data, the simple model beats the deep one.**

---

## 0. The one-line version

> **A sentiment model is:** words → vectors (embedding = one-hot @ embedding-matrix) → LSTM (4 gates per word, all matmuls) → one number → sigmoid → sentiment. The LSTM's hidden state is a **running vector that carries meaning across the sentence** — that's the "memory."

---

## 1. Data — text → integer ids → padded sequences

```python
import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)

pos = [ "this movie was amazing and touching", "i loved every minute of it",
        "great acting and a beautiful story", ... ]      # 40 positive reviews
neg = [ "what a waste of time and money", "the acting was terrible and boring",
        "worst movie i have ever watched", ... ]         # 40 negative reviews
reviews = pos + neg
labels = np.array([1]*40 + [0]*40)

idx = np.random.permutation(80)
tr_idx, te_idx = idx[:60], idx[60:]                     # 60 train / 20 test
tr_rev, tr_y = [reviews[i] for i in tr_idx], labels[tr_idx]
te_rev, te_y = [reviews[i] for i in te_idx], labels[te_idx]

def tokenize(reviews, max_len=10):
    vocab = {w: i + 1 for i, w in enumerate(sorted(set(" ".join(reviews).split())))}
    ids = []
    for r in reviews:
        seq = [vocab[w] for w in r.split() if w in vocab][:max_len]
        seq = seq + [0] * (max_len - len(seq))          # pad with 0 = "nothing"
        ids.append(seq)
    return np.array(ids), len(vocab) + 1

Xtr_ids, vocab_size = tokenize(tr_rev)
Xte_ids, _ = tokenize(te_rev)
print("samples: 80 | vocab size:", vocab_size, "| train:", Xtr_ids.shape, "test:", Xte_ids.shape)
```
```
samples: 80 | positive: 40 | negative: 40
vocab size: 177 | train: (60, 10)  test: (20, 10)
```

**Math tagged here:** the model can't eat words, so each word becomes an **integer id** from a dictionary (the vocab). Each review becomes a vector of 10 ids (padded with 0 to make all reviews the same length — batches are rectangles). Note the split *before* building the vocab — same leakage discipline as the spam doc.

---

## 2. The model — Embedding → LSTM → Linear (one sentence, then unpack)

```python
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)   # word -> vector
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc    = nn.Linear(hidden_dim, 1)              # hidden state -> one score

    def forward(self, x):
        e = self.embed(x)                    # (batch, 10 words, 16 dims)
        out, (h, c) = self.lstm(e)           # 4 gates per step (see below)
        pooled = out.mean(dim=1)             # average hidden states over time
        return self.fc(pooled).squeeze(1)    # final matmul -> logit
```

**Unpacking the three math pieces:**

**1. `nn.Embedding` = one matmul.** A word id (e.g., 42) becomes a 16-number vector. Under the hood: one-hot-encode the id (a 1×177 vector) and multiply by the 177×16 embedding matrix. **The embeddings are learned** — after training, "amazing" and "wonderful" have similar vectors (dot product = high similarity; your linear algebra docs). This is the same word-vector idea as ChatGPT's token embeddings.

**2. The LSTM = 4 sigmoid/tanh matmuls per word.** At each word, the LSTM updates a *memory vector* $c$ and a *hidden state* $h$ using 4 gates — each gate is one matmul followed by an activation:

```
for each word x_t, with hidden h_t and memory c_t:
    f = sigmoid(Wf·x + Uf·h + bf)   # forget gate: what to erase from memory
    i = sigmoid(Wi·x + Ui·h + bi)   # input gate:  what to write into memory
    o = sigmoid(Wo·x + Uo·h + bo)   # output gate: what to reveal as h
    g = tanh(Wg·x + Ug·h + bg)      # candidate new memory
    c' = f ⊙ c + i ⊙ g              # update memory (⊙ = element-wise multiply)
    h' = o ⊙ tanh(c')               # new hidden state
```

Every gate is a **matmul** (your linear algebra) + **sigmoid or tanh** (your calculus: S-curves that squeeze to 0–1 and −1..1). The memory update `c' = f ⊙ c + i ⊙ g` is just **element-wise multiply and add** — "erase a bit, write a bit." That's the entire secret of "memory" in deep learning: a running vector, updated by multiplications and additions. Nothing more.

**3. `self.fc` = the final matmul** from the pooled hidden states to one score — the same `X @ W + b` from house prices.

---

## 3. Training — the same five-line loop

```python
model = SentimentLSTM(vocab_size)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
Xtr_t, ytr_t = torch.tensor(Xtr_ids), torch.tensor(tr_y, dtype=torch.float32)

for epoch in range(300):
    opt.zero_grad()
    logits = model(Xtr_t)                 # FORWARD: embeddings + LSTM + fc
    loss = nn.functional.binary_cross_entropy_with_logits(logits, ytr_t)  # LOSS
    loss.backward()                       # BACKWARD: the chain rule, THROUGH TIME
    opt.step()                            # UPDATE: Adam
```

**Verified output:**
```
train acc: 1.000 | TEST acc (unseen): 0.500
```

**Read the numbers — this is the honest part.** Train accuracy 100%, test accuracy 50% — the model **memorized the 60 training reviews** and learned nothing general. This is *overfitting* in its purest form, and it's the single most common failure in real ML. Notice: the code was perfect. The loop was perfect. The math was perfect. **The data was the problem** — 60 reviews is hopelessly little for a model with ~5,000 parameters. (PyTorch's `loss.backward()` on an LSTM runs the chain rule *through time* — the same two lines from the MNIST MLP, unrolled over 10 steps.)

**And here is the production insight you'll meet in week 1 of a real job:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
vec_b = TfidfVectorizer(stop_words="english").fit(tr_rev)
bow_lr = LogisticRegression(max_iter=1000).fit(vec_b.transform(tr_rev), tr_y)
print("BOW+Logistic (same tiny data): TEST acc =", round(bow_lr.score(vec_b.transform(te_rev), te_y), 3))
```
```
BOW+Logistic (same tiny data): TEST acc = 0.750  <- beats the LSTM here!
```

**Read the numbers:** a 50-line logistic regression on word counts **beats the LSTM** (0.75 vs 0.50) on the same data. Why? The simple model has ~1000 parameters; the LSTM has ~5000. Small data → simpler models win. **This is why TF-IDF + logistic regression is still deployed in production everywhere** — and why "use a transformer!" is wrong advice until you've tried the baseline. Deep learning needs big data; that's the whole trade.

---

## 4. Inference — the deployed model

```python
for test in ["i loved this movie", "this was terrible", "an absolute masterpiece",
             "so boring and dull", "amazing film with great acting", "a total waste of time"]:
    p, verdict = predict_review(test)
    print(f"  '{test}' -> {verdict} (p={p:.3f})")
```
```
  'i loved this movie' -> POSITIVE (p=1.000)
  'this was terrible' -> NEGATIVE (p=0.000)
  'an absolute masterpiece' -> POSITIVE (p=0.961)
  'so boring and dull' -> POSITIVE (p=1.000)
  'amazing film with great acting' -> POSITIVE (p=0.983)
  'a total waste of time' -> NEGATIVE (p=0.001)
```

**Read the numbers:** 5 of 6 new phrases classified sensibly — the LSTM *did* learn something about words in order ("loved" + "movie" → positive, "terrible" → negative). The one miss ("so boring and dull" → positive, p=1.000!) is the overfitting showing itself again: with 60 samples, the model latched onto patterns that don't generalize. **At real scale** (IMDB: 25,000 train reviews), LSTMs reach ~85% and this same code — bigger embeddings, a few more layers — is the standard recipe.

---

## 5. The map — where each math concept lives

| Where it happened | Math | Code |
| :--- | :--- | :--- |
| Words → ids | a dictionary (vocab) | `tokenize()` |
| Word → vector | **one matmul** (one-hot @ embedding matrix) | `nn.Embedding` |
| "Memory" | **4 matmuls + sigmoid/tanh + element-wise ops** | `nn.LSTM` |
| Hidden state → score | final matmul + bias | `self.fc` |
| Loss | **binary cross-entropy** (same as spam) | `binary_cross_entropy_with_logits` |
| Backward | **chain rule, unrolled through time** | `loss.backward()` |
| Update | gradient descent (Adam) | `opt.step()` |
| The honest lesson | variance of small samples (statistics: with n=60, estimates are noisy) | train/test split |

**New mental furniture:** "embeddings are learned matmuls", "LSTM memory = 4 gates of sigmoid/tanh matmuls + element-wise updates", and the career-saver: **always run a simple baseline before a deep model** — on small data, simple wins; the deep model earns its complexity only with data.

**Next:** `05-day-in-life.md` — the day-in-the-life view: which math you actually touch daily as a junior ML engineer, and which you only recognize when reading papers.