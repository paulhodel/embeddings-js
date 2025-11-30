# Model 1: Position-Weighted Context-to-Vector Prediction

## Overview

This document describes the complete algorithm for Model 1, a novel word embedding approach that learns embeddings through direct vector prediction from position-weighted context, without classification or negative sampling.

## Key Characteristics

- **Training Objective**: Regression (predict target embedding from context)
- **No Softmax**: Avoids expensive 100K classification layer
- **Deep Network**: Two hidden layers (1024 → 1024) for complex transformations
- **Position-Aware**: Context words weighted by distance from target
- **Target-Only Updates**: Only update target word embeddings, not context (for stability)
- **Vocabulary**: 50K most common words

## Architecture Summary

```
Input: Weighted average of context embeddings (1024 floats)
  ↓ W1 [1024 × 1024]
Hidden Layer 1: 1024 neurons (ReLU activation)
  ↓ W2 [1024 × 1024]
Hidden Layer 2: 1024 neurons (ReLU activation)
  ↓ (Output = Hidden Layer 2)
Output: Predicted target embedding (1024 floats)

Loss: MSE(predicted_embedding, actual_target_embedding)

Updates:
  - W1, W2 (network weights)
  - Target word embedding only
  - Context word embeddings NOT updated (they update when they become targets)
```

---

## Phase 1: Vocabulary Preparation (prepare.js)

### Input
- Parquet files from `./data/parquet/` (downloaded by download.js)
- Each parquet file contains text data from fineweb-edu corpus

### Process

#### Step 1: Extract Text from All Parquet Files
```javascript
// Read all .parquet files from ./data/parquet/
// Extract text field from each row
// Concatenate all text into one large corpus
```

#### Step 2: Tokenization
```javascript
// Simple whitespace + punctuation tokenization
// Convert to lowercase
// Examples:
//   "The cat sat." → ["the", "cat", "sat", "."]
//   "Hello, world!" → ["hello", ",", "world", "!"]
```

#### Step 3: Count Word Frequencies
```javascript
const wordCounts = {};
for (const token of allTokens) {
    wordCounts[token] = (wordCounts[token] || 0) + 1;
}
```

#### Step 4: Select Top 50K Words
```javascript
// Sort by frequency (descending)
// Take top 50,000 words
// This typically covers ~95% of tokens in the corpus
```

#### Step 5: Initialize Dictionary
```javascript
const dictionary = {};
const embeddingDim = 1024;

for (let i = 0; i < topWords.length; i++) {
    const word = topWords[i];
    dictionary[word] = {
        id: i,
        vector: initializeVector(embeddingDim),
        frequency: wordCounts[word]
    };
}

function initializeVector(dim) {
    // Random initialization: uniform distribution [-0.1, 0.1]
    const vector = [];
    for (let i = 0; i < dim; i++) {
        vector[i] = (Math.random() * 0.2) - 0.1;
    }
    return vector;
}
```

#### Step 6: Calculate OOV Statistics
```javascript
// Count how many tokens are in vocabulary vs OOV
// Report:
//   - Vocabulary size: 50,000
//   - Total tokens in corpus: X
//   - Tokens covered: Y (Z%)
//   - OOV tokens: X-Y (Z%)
//   - Estimated training windows lost: ~W%
```

### Output
**File: `./data/dictionary.json`**
```json
{
  "the": {
    "id": 0,
    "vector": [0.023, -0.145, 0.089, ..., 0.034],
    "frequency": 5234891
  },
  "cat": {
    "id": 42,
    "vector": [-0.112, 0.234, -0.067, ..., 0.156],
    "frequency": 12453
  },
  ...
}
```

**File: `./data/vocab_stats.json`**
```json
{
  "vocabularySize": 50000,
  "totalTokens": 100000000,
  "tokensCovered": 95234567,
  "coveragePercent": 95.23,
  "oovTokens": 4765433,
  "oovPercent": 4.77,
  "estimatedWindowLoss": 15.2
}
```

---

## Phase 2: Training (train.js)

### Hyperparameters

```javascript
const EMBEDDING_DIM = 1024;
const HIDDEN_DIM = 1024;
const CONTEXT_WINDOW_SIZE = 3;  // 3 words on each side of target
const LEARNING_RATE = 0.01;     // To be tuned
const BATCH_SIZE = 100;         // Mini-batch gradient descent
const NUM_EPOCHS = 5;           // Multiple passes through corpus

// Position weights (closer = more important)
const POSITION_WEIGHTS = {
    '-3': 1/3,
    '-2': 1/2,
    '-1': 1,
    '+1': 1,
    '+2': 1/2,
    '+3': 1/3
};

// Quality thresholds for partial context
const MIN_CONTEXT_WORDS = 4;  // Need at least 4 out of 6 context words
```

### Network Initialization

```javascript
// Weight matrices initialized with He initialization (good for ReLU)
function initializeWeights(inputSize, outputSize) {
    const weights = [];
    const std = Math.sqrt(2.0 / inputSize);  // He initialization

    for (let i = 0; i < outputSize; i++) {
        weights[i] = [];
        for (let j = 0; j < inputSize; j++) {
            // Box-Muller transform for normal distribution
            const u1 = Math.random();
            const u2 = Math.random();
            const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            weights[i][j] = z * std;
        }
    }
    return weights;
}

let W1 = initializeWeights(1024, 1024);  // Input → Hidden1
let W2 = initializeWeights(1024, 1024);  // Hidden1 → Hidden2
```

### Training Loop Structure

```javascript
// Load dictionary
const dictionary = JSON.parse(fs.readFileSync('./data/dictionary.json'));

// Load and tokenize corpus
const corpus = loadAndTokenizeCorpus('./data/parquet/');

for (let epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    console.log(`Epoch ${epoch + 1}/${NUM_EPOCHS}`);

    // Batch accumulators
    let batchGradients = {
        W1: createZeroMatrix(1024, 1024),
        W2: createZeroMatrix(1024, 1024),
        embeddings: {}  // word → accumulated gradient
    };
    let batchCount = 0;
    let batchLoss = 0;

    // Iterate through corpus with sliding window
    for (let position = 0; position < corpus.length; position++) {
        // Extract window (may return null if quality checks fail)
        const window = extractWindow(corpus, position, dictionary);

        if (window === null) {
            continue;  // Skip this position
        }

        // Forward pass
        const predicted = forwardPass(window, dictionary, W1, W2);

        // Calculate loss
        const target = dictionary[window.target].vector;
        const loss = calculateMSE(predicted, target);
        batchLoss += loss;

        // Backward pass (accumulate gradients)
        const gradients = backwardPass(predicted, target, window, dictionary, W1, W2);

        // Accumulate
        accumulateGradients(batchGradients, gradients, window);
        batchCount++;

        // Update after batch
        if (batchCount === BATCH_SIZE) {
            applyBatchUpdate(batchGradients, batchCount, LEARNING_RATE, dictionary, W1, W2);

            // Log progress
            const avgLoss = batchLoss / batchCount;
            console.log(`  Position ${position}/${corpus.length}, Avg Loss: ${avgLoss.toFixed(6)}`);

            // Reset batch
            batchGradients = {
                W1: createZeroMatrix(1024, 1024),
                W2: createZeroMatrix(1024, 1024),
                embeddings: {}
            };
            batchCount = 0;
            batchLoss = 0;
        }
    }

    // Update remaining batch (if any)
    if (batchCount > 0) {
        applyBatchUpdate(batchGradients, batchCount, LEARNING_RATE, dictionary, W1, W2);
    }

    // Save checkpoint after each epoch
    saveCheckpoint(dictionary, W1, W2, epoch);
}
```

### Window Extraction (Partial Context with Quality Threshold)

```javascript
/**
 * Extract training window from corpus at given position
 * Returns null if quality checks fail
 */
function extractWindow(corpus, position, dictionary) {
    const contextSize = 3;
    const targetWord = corpus[position];

    // CHECK 1: Target must be in dictionary
    if (!dictionary[targetWord]) {
        return null;  // Skip if target is OOV
    }

    // Collect available context words
    const contextWords = [];
    const contextPositions = [];

    for (let offset = -contextSize; offset <= contextSize; offset++) {
        if (offset === 0) continue;  // Skip target position

        const pos = position + offset;

        // Check bounds
        if (pos < 0 || pos >= corpus.length) {
            continue;
        }

        const word = corpus[pos];

        // Check if word is in dictionary
        if (dictionary[word]) {
            contextWords.push(word);
            contextPositions.push(offset);
        }
    }

    // CHECK 2: Need at least 4 out of 6 context words
    if (contextWords.length < 4) {
        return null;
    }

    // CHECK 3: Must have at least one word from each side
    const hasLeftContext = contextPositions.some(p => p < 0);
    const hasRightContext = contextPositions.some(p => p > 0);

    if (!hasLeftContext || !hasRightContext) {
        return null;
    }

    // CHECK 4 (Optional): Prefer balanced context
    // If missing both close words on one side (-1,-2 or +1,+2), consider skipping
    // This is optional - can be removed if too strict
    const hasLeftClose = contextPositions.some(p => p === -1 || p === -2);
    const hasRightClose = contextPositions.some(p => p === 1 || p === 2);

    if (!hasLeftClose || !hasRightClose) {
        // Still allow, but this is a quality indicator
        // Could skip here if you want higher quality
    }

    return {
        target: targetWord,
        contextWords: contextWords,
        contextPositions: contextPositions
    };
}
```

### Forward Pass

```javascript
/**
 * Forward propagation through the network
 */
function forwardPass(window, dictionary, W1, W2) {
    // Step 1: Get context embeddings
    const contextEmbeddings = window.contextWords.map(word =>
        dictionary[word].vector
    );

    // Step 2: Compute weighted average
    const input = computeWeightedAverage(
        contextEmbeddings,
        window.contextPositions
    );

    // Step 3: First hidden layer
    const z1 = matmul(input, W1);  // [1024] × [1024×1024] → [1024]
    const a1 = z1.map(x => relu(x));  // ReLU activation

    // Step 4: Second hidden layer
    const z2 = matmul(a1, W2);  // [1024] × [1024×1024] → [1024]
    const a2 = z2.map(x => relu(x));  // ReLU activation

    // Step 5: Output (no activation, this is regression)
    const predicted = a2;

    // Store activations for backward pass
    return {
        predicted: predicted,
        activations: {
            input: input,
            z1: z1,
            a1: a1,
            z2: z2,
            a2: a2
        }
    };
}

/**
 * Compute weighted average of context embeddings
 */
function computeWeightedAverage(embeddings, positions) {
    const positionWeights = {
        '-3': 1/3, '-2': 1/2, '-1': 1,
        '+1': 1, '+2': 1/2, '+3': 1/3
    };

    const dim = embeddings[0].length;  // 1024
    const weightedSum = new Array(dim).fill(0);
    let totalWeight = 0;

    for (let i = 0; i < embeddings.length; i++) {
        const weight = positionWeights[positions[i].toString()];
        totalWeight += weight;

        for (let j = 0; j < dim; j++) {
            weightedSum[j] += embeddings[i][j] * weight;
        }
    }

    // Normalize by total weight
    return weightedSum.map(x => x / totalWeight);
}

/**
 * ReLU activation function
 */
function relu(x) {
    return Math.max(0, x);
}

/**
 * Matrix-vector multiplication
 */
function matmul(vector, matrix) {
    // vector: [1024]
    // matrix: [1024][1024]
    // result: [1024]
    const result = [];

    for (let i = 0; i < matrix.length; i++) {
        let sum = 0;
        for (let j = 0; j < vector.length; j++) {
            sum += vector[j] * matrix[i][j];
        }
        result[i] = sum;
    }

    return result;
}
```

### Loss Calculation

```javascript
/**
 * Mean Squared Error loss
 */
function calculateMSE(predicted, target) {
    let sum = 0;
    const n = predicted.length;  // 1024

    for (let i = 0; i < n; i++) {
        const diff = predicted[i] - target[i];
        sum += diff * diff;
    }

    return sum / n;
}
```

### Backward Pass

```javascript
/**
 * Backpropagation to compute gradients
 */
function backwardPass(forwardResult, target, window, dictionary, W1, W2) {
    const predicted = forwardResult.predicted;
    const { input, z1, a1, z2, a2 } = forwardResult.activations;
    const n = predicted.length;  // 1024

    // Gradient of MSE loss: dL/dpredicted = 2 * (predicted - target) / n
    const dLoss = [];
    for (let i = 0; i < n; i++) {
        dLoss[i] = 2 * (predicted[i] - target[i]) / n;
    }

    // Backprop through second layer
    // dL/dz2 = dLoss * relu'(z2)
    const dz2 = [];
    for (let i = 0; i < n; i++) {
        dz2[i] = z2[i] > 0 ? dLoss[i] : 0;  // ReLU derivative
    }

    // dL/dW2 = a1^T * dz2 (outer product)
    const dW2 = [];
    for (let i = 0; i < W2.length; i++) {
        dW2[i] = [];
        for (let j = 0; j < W2[i].length; j++) {
            dW2[i][j] = a1[j] * dz2[i];
        }
    }

    // dL/da1 = W2^T * dz2
    const da1 = new Array(a1.length).fill(0);
    for (let i = 0; i < a1.length; i++) {
        for (let j = 0; j < dz2.length; j++) {
            da1[i] += W2[j][i] * dz2[j];
        }
    }

    // Backprop through first layer
    // dL/dz1 = da1 * relu'(z1)
    const dz1 = [];
    for (let i = 0; i < n; i++) {
        dz1[i] = z1[i] > 0 ? da1[i] : 0;  // ReLU derivative
    }

    // dL/dW1 = input^T * dz1 (outer product)
    const dW1 = [];
    for (let i = 0; i < W1.length; i++) {
        dW1[i] = [];
        for (let j = 0; j < W1[i].length; j++) {
            dW1[i][j] = input[j] * dz1[i];
        }
    }

    // dL/dinput = W1^T * dz1
    const dInput = new Array(input.length).fill(0);
    for (let i = 0; i < input.length; i++) {
        for (let j = 0; j < dz1.length; j++) {
            dInput[i] += W1[j][i] * dz1[j];
        }
    }

    // Backprop through weighted average to get gradients for embeddings
    // Only compute gradient for TARGET embedding (not context)
    const positionWeights = {
        '-3': 1/3, '-2': 1/2, '-1': 1,
        '+1': 1, '+2': 1/2, '+3': 1/3
    };

    let totalWeight = 0;
    for (const pos of window.contextPositions) {
        totalWeight += positionWeights[pos.toString()];
    }

    // Gradient for target embedding (direct gradient from loss)
    const dTarget = dLoss.slice();  // Copy of loss gradient

    // NOTE: We do NOT compute gradients for context embeddings
    // They will be updated when they appear as targets

    return {
        dW1: dW1,
        dW2: dW2,
        dTarget: dTarget
    };
}
```

### Gradient Accumulation and Update

```javascript
/**
 * Accumulate gradients for mini-batch
 */
function accumulateGradients(batchGradients, gradients, window) {
    const { dW1, dW2, dTarget } = gradients;

    // Accumulate network gradients
    for (let i = 0; i < dW1.length; i++) {
        for (let j = 0; j < dW1[i].length; j++) {
            batchGradients.W1[i][j] += dW1[i][j];
        }
    }

    for (let i = 0; i < dW2.length; i++) {
        for (let j = 0; j < dW2[i].length; j++) {
            batchGradients.W2[i][j] += dW2[i][j];
        }
    }

    // Accumulate target embedding gradient
    const targetWord = window.target;
    if (!batchGradients.embeddings[targetWord]) {
        batchGradients.embeddings[targetWord] = new Array(1024).fill(0);
    }

    for (let i = 0; i < dTarget.length; i++) {
        batchGradients.embeddings[targetWord][i] += dTarget[i];
    }
}

/**
 * Apply accumulated gradients (mini-batch update)
 */
function applyBatchUpdate(batchGradients, batchSize, learningRate, dictionary, W1, W2) {
    // Update W1
    for (let i = 0; i < W1.length; i++) {
        for (let j = 0; j < W1[i].length; j++) {
            const gradient = batchGradients.W1[i][j] / batchSize;
            W1[i][j] -= learningRate * gradient;
        }
    }

    // Update W2
    for (let i = 0; i < W2.length; i++) {
        for (let j = 0; j < W2[i].length; j++) {
            const gradient = batchGradients.W2[i][j] / batchSize;
            W2[i][j] -= learningRate * gradient;
        }
    }

    // Update target embeddings (ONLY targets, not context)
    for (const word in batchGradients.embeddings) {
        const gradient = batchGradients.embeddings[word];
        for (let i = 0; i < gradient.length; i++) {
            const avgGradient = gradient[i] / batchSize;
            dictionary[word].vector[i] -= learningRate * avgGradient;
        }
    }
}
```

### Monitoring and Evaluation

During training, monitor:

1. **Loss**: Should decrease over time
2. **Embedding Variance**: Should NOT collapse to zero (all embeddings becoming identical)
3. **Sample Similarities**: Check if semantically similar words are getting similar embeddings

```javascript
/**
 * Calculate cosine similarity between two vectors
 */
function cosineSimilarity(vec1, vec2) {
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < vec1.length; i++) {
        dotProduct += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

/**
 * Check embedding quality during training
 */
function evaluateEmbeddings(dictionary) {
    // Calculate variance of all embeddings
    const allEmbeddings = Object.values(dictionary).map(d => d.vector);
    const variance = calculateVariance(allEmbeddings);

    console.log(`  Embedding variance: ${variance.toFixed(6)}`);

    // Check sample similarities
    const testPairs = [
        ["king", "queen"],
        ["man", "woman"],
        ["cat", "dog"],
        ["good", "bad"]
    ];

    console.log("  Sample similarities:");
    for (const [word1, word2] of testPairs) {
        if (dictionary[word1] && dictionary[word2]) {
            const sim = cosineSimilarity(
                dictionary[word1].vector,
                dictionary[word2].vector
            );
            console.log(`    ${word1} - ${word2}: ${sim.toFixed(4)}`);
        }
    }
}
```

### Checkpointing

```javascript
/**
 * Save training checkpoint
 */
function saveCheckpoint(dictionary, W1, W2, epoch) {
    const checkpoint = {
        epoch: epoch,
        dictionary: dictionary,
        W1: W1,
        W2: W2,
        timestamp: new Date().toISOString()
    };

    const filename = `./data/checkpoint_epoch${epoch}.json`;
    fs.writeFileSync(filename, JSON.stringify(checkpoint));
    console.log(`Checkpoint saved: ${filename}`);
}

/**
 * Load training checkpoint
 */
function loadCheckpoint(filename) {
    const checkpoint = JSON.parse(fs.readFileSync(filename));
    return {
        dictionary: checkpoint.dictionary,
        W1: checkpoint.W1,
        W2: checkpoint.W2,
        startEpoch: checkpoint.epoch + 1
    };
}
```

---

## Phase 3: Evaluation and Benchmarking

### Analogy Task

Test: "king - man + woman ≈ queen"

```javascript
/**
 * Vector arithmetic for analogies
 */
function solveAnalogy(wordA, wordB, wordC, dictionary, topK = 5) {
    // Compute: wordB - wordA + wordC
    // Find nearest word to result

    const vecA = dictionary[wordA].vector;
    const vecB = dictionary[wordB].vector;
    const vecC = dictionary[wordC].vector;

    // Compute target vector
    const target = [];
    for (let i = 0; i < vecA.length; i++) {
        target[i] = vecB[i] - vecA[i] + vecC[i];
    }

    // Find nearest neighbors (excluding input words)
    const candidates = [];
    for (const word in dictionary) {
        if (word === wordA || word === wordB || word === wordC) continue;

        const similarity = cosineSimilarity(target, dictionary[word].vector);
        candidates.push({ word, similarity });
    }

    // Sort by similarity (descending)
    candidates.sort((a, b) => b.similarity - a.similarity);

    return candidates.slice(0, topK);
}

// Example test
const result = solveAnalogy("king", "man", "woman", dictionary);
console.log("king - man + woman ≈", result[0].word);  // Expect "queen"
```

### Similarity Task

Test semantic similarity:

```javascript
/**
 * Find most similar words to a given word
 */
function findSimilar(word, dictionary, topK = 10) {
    const targetVec = dictionary[word].vector;

    const candidates = [];
    for (const candidateWord in dictionary) {
        if (candidateWord === word) continue;

        const similarity = cosineSimilarity(targetVec, dictionary[candidateWord].vector);
        candidates.push({ word: candidateWord, similarity });
    }

    candidates.sort((a, b) => b.similarity - a.similarity);
    return candidates.slice(0, topK);
}

// Example
console.log("Words similar to 'cat':", findSimilar("cat", dictionary));
```

---

## Expected Challenges and Mitigation

### Challenge 1: Circular Learning (Embeddings Bootstrap from Random)

**Problem**: Early in training, all embeddings are random. Network predicts random values, embeddings updated toward random predictions.

**Why it should resolve**:
- Words in similar contexts receive similar predictions over many iterations
- Gradual convergence toward meaningful structure
- Co-occurrence patterns emerge

**Mitigation**:
- Multiple epochs (5-10)
- Monitor embedding variance (should not collapse)
- Patience - may take time to converge

### Challenge 2: Mode Collapse (All Embeddings Become Similar)

**Problem**: All embeddings might converge to similar values.

**Detection**: Monitor embedding variance during training

**Mitigation**:
- Learning rate tuning
- L2 regularization (optional)
- Ensure diverse training data

### Challenge 3: High-Frequency Word Dominance

**Problem**: Common words like "the", "a" appear much more frequently, might dominate learning.

**Mitigation**:
- Mini-batch gradient descent averages gradients
- Consider subsampling frequent words (optional)
- Monitor per-word update counts

### Challenge 4: Slow Convergence

**Problem**: Training might take many epochs to converge.

**Expectation**:
- 5-10 epochs on 5 shards with 50K vocabulary
- Several hours of training time

**Optimization**:
- Batch processing
- Efficient matrix operations
- Consider GPU if available

---

## Success Criteria

### Minimum Viable Success
- Loss decreases consistently over epochs
- Embedding variance remains stable (not collapsing)
- Some semantic similarity emerges (e.g., "cat" closer to "dog" than "quantum")

### Good Success
- Clear semantic clusters (animals, colors, actions, etc.)
- Simple analogies work (king/queen, man/woman)
- Achieves 60-80% of Word2Vec quality on benchmarks

### Excellent Success
- Complex analogies work
- Achieves comparable quality to Word2Vec
- Demonstrates that vector prediction is viable alternative to classification

---

## Implementation Checklist

### prepare.js
- [ ] Read parquet files
- [ ] Tokenize text
- [ ] Count word frequencies
- [ ] Select top 50K words
- [ ] Initialize random embeddings
- [ ] Calculate OOV statistics
- [ ] Save dictionary.json

### train.js
- [ ] Load dictionary
- [ ] Initialize network weights (W1, W2)
- [ ] Implement window extraction with quality checks
- [ ] Implement weighted average computation
- [ ] Implement forward pass
- [ ] Implement MSE loss
- [ ] Implement backward pass
- [ ] Implement mini-batch gradient accumulation
- [ ] Implement weight updates (target-only for embeddings)
- [ ] Add progress logging
- [ ] Add evaluation during training
- [ ] Add checkpointing
- [ ] Multiple epochs support

### utils.js
- [ ] Matrix operations (matmul, zeros)
- [ ] Activation functions (ReLU)
- [ ] Loss functions (MSE)
- [ ] Similarity metrics (cosine)
- [ ] He initialization
- [ ] Helper functions

### evaluate.js
- [ ] Analogy tests
- [ ] Similarity tests
- [ ] Visualization tools
- [ ] Benchmark comparisons

---

## File Structure

```
src/
├── prepare.js          # Vocabulary preparation
├── train.js            # Training loop
├── evaluate.js         # Evaluation and benchmarks
└── utils.js            # Helper functions

data/
├── parquet/            # Downloaded corpus files
├── dictionary.json     # Vocabulary with embeddings
├── vocab_stats.json    # Coverage statistics
├── checkpoint_*.json   # Training checkpoints
└── weights/
    ├── W1.json         # Network weights
    └── W2.json

results/
├── training_log.txt    # Training progress
└── evaluation.json     # Benchmark results
```

---

## Research Questions for Model 2 Comparison

Once Model 1 is working, Model 2 will explore different approaches:

1. Update both context and target embeddings (vs target-only)
2. Different architectures (negative sampling, hierarchical softmax, etc.)
3. Different context aggregation (learned attention vs fixed weights)
4. Classification vs regression objectives

The comparison will reveal:
- Speed vs quality trade-offs
- Stability characteristics
- When each approach is preferred

---

## Notes on Code Style

Following the reference implementation patterns:
- Manual loops for matrix operations (performance)
- Explicit variable names (clarity)
- JSDoc comments for all functions
- Mathematical operations clearly commented
- Academic-style documentation

---

**Last Updated**: 2025-11-29
**Status**: Ready for implementation
