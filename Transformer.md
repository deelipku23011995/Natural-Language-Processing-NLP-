# Self-Attention in Transformers

Self-attention enables each word in a sentence to focus on the most relevant words, allowing the model to capture contextual relationships regardless of their position.

## Step 1: Generate Query, Key, and Value

Each input embedding **X** is projected into three vectors using learned weight matrices:

* **Query (Q)** = X × WQ
* **Key (K)** = X × WK
* **Value (V)** = X × WV

Typically, embeddings have a higher dimension (e.g., 512), while Q, K, and V use a smaller dimension (e.g., 64) to reduce computation.

## Step 2: Compute Attention Scores

For each word, calculate the similarity between its **Query** and the **Keys** of all words using the dot product:

```text
Score = Q × Kᵀ
```

Higher scores indicate stronger relevance between words.

## Step 3: Scale the Scores

Normalize the scores by dividing them by the square root of the key dimension:

```text
Scaled Score = Score / √dk
```

This improves numerical stability and helps maintain stable gradients during training.

## Step 4: Apply Softmax

Convert the scaled scores into probabilities:

```text
Attention Weights = Softmax(Scaled Score)
```

The resulting attention weights:

* Are positive
* Sum to 1
* Represent how much attention each word receives

## Step 5: Weight the Values

Multiply each **Value** vector by its corresponding attention weight. More relevant words contribute more to the final representation.

## Step 6: Compute the Output

Sum the weighted Value vectors to obtain the final self-attention output:

```text
Output = Σ (Attention Weight × Value)
```

The output is then passed to the feed-forward network.

---

# Matrix Formulation

Instead of processing each word individually, all words are processed simultaneously.

```text
Q = XWQ
K = XWK
V = XWV
```

The complete self-attention operation is:

```text
Attention(Q, K, V) = Softmax((QKᵀ) / √dk) V
```

This single equation performs:

1. Similarity computation
2. Score scaling
3. Softmax normalization
4. Weighted aggregation of Value vectors

---

# Key Concepts

| Component     | Purpose                                             |
| ------------- | --------------------------------------------------- |
| **Query (Q)** | Represents what the current word is searching for   |
| **Key (K)**   | Represents what information each word contains      |
| **Value (V)** | Represents the information passed to the next layer |
| **Softmax**   | Converts scores into attention probabilities        |
| **Output**    | Context-aware representation of each input word     |

