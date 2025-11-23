# Polysemy Analysis Guide

## Overview

The **PolysemyAnalyzer** verifies that CSS is learning multiple senses for polysemous words by analyzing **sense separation** in the learned spectra.

---

## What It Does

For a polysemous word like "bank":

1. **Collects contexts** where the word appears
2. **Builds context spectra** (sum of context word spectra, excluding target)
3. **Clusters contexts** using k-means (e.g., financial vs. river)
4. **Analyzes word spectrum** to see if different frequencies align with different clusters
5. **Delivers verdict** on whether polysemy has emerged

---

## Quick Start

```bash
# Run the demo analysis
npm run analyze
```

This analyzes the word "bank" in a sample corpus with both financial and river senses.

---

## Example Output

```
======================================================================
POLYSEMY ANALYSIS: "bank"
======================================================================

Step 1: Collecting contexts...
  Found 19 occurrences of "bank"

Step 2: Building context spectra...
  Built 19 context spectra

Step 3: Clustering into 2 sense clusters...
  Converged after 3 iterations

  Cluster 1: 9 contexts
    Top frequencies: 26(0.299), 2(0.251), 1(0.227)
    Sample contexts:
      - [...the offers loans with...]
      - [...at the financial downtown...]

  Cluster 2: 10 contexts
    Top frequencies: 5(0.364), 26(0.235), 36(0.210)
    Sample contexts:
      - [...the river is covered...]
      - [...sat on the and watched...]

Step 4: Analyzing target word spectrum...
  "bank" has 4 active frequencies

  Dominant frequencies:
    1. Freq=2, Amp=0.3701
    2. Freq=46, Amp=0.3287
    3. Freq=47, Amp=0.0479
    4. Freq=34, Amp=0.0337

Step 5: Checking sense-cluster alignment...

  Sense 1 (Cluster 1):
    Alignment score: 0.3213
    Overlapping frequencies: 2, 34, 46

  Sense 2 (Cluster 2):
    Alignment score: 0.0745
    Overlapping frequencies: 46, 47

----------------------------------------------------------------------
VERDICT:
----------------------------------------------------------------------
✅ POLYSEMY DETECTED!
   Word has 4 active frequencies
   2 distinct context clusters found
   Alignment spread: 0.247
   Frequencies appear to separate senses
----------------------------------------------------------------------
```

---

## How It Works

### Step 1: Collect Contexts

For each occurrence of the target word:
- Extract surrounding words (window size = 3)
- Exclude the target word itself
- Store as context

Example for "bank":
```
Financial: [...lends, money, to, businesses...]
River:     [...sat, on, and, watched, water...]
```

### Step 2: Build Context Spectra

For each context:
- Sum the spectra of all context words
- Normalize to unit length
- Result: a "context signature" in frequency space

```javascript
contextSpectrum = Σ (context word spectra)
```

### Step 3: Cluster Contexts

Use k-means clustering (k=2 for two senses):
- Group similar context spectra together
- Financial contexts cluster together
- River contexts cluster together

Each cluster centroid represents the "typical frequency signature" of that sense.

### Step 4: Analyze Word Spectrum

Check the target word's learned spectrum:
- How many active frequencies does it have?
- Which frequencies are dominant?

Example:
```
Word "bank": [Freq 2: 0.37, Freq 46: 0.33, Freq 47: 0.05, Freq 34: 0.03]
```

### Step 5: Check Alignment

Calculate alignment between word frequencies and cluster centroids:

**Good polysemy:**
- Cluster 1 aligns with Freq 2, Freq 34
- Cluster 2 aligns with Freq 46, Freq 47
- Different frequencies → different senses

**No polysemy:**
- All clusters align with same frequency
- Or word has only 1 frequency

---

## Interpretation

### ✅ Polysemy Detected
```
✅ POLYSEMY DETECTED!
   Word has 4 active frequencies
   2 distinct context clusters found
   Alignment spread: 0.247
   Frequencies appear to separate senses
```

**Meaning:**
- Word has multiple active frequencies (4)
- Contexts cluster into distinct groups (2)
- Different frequencies align with different clusters
- **CSS successfully learned multiple senses!**

### ⚠️ Partial Polysemy
```
⚠️  PARTIAL POLYSEMY
   Word has 2 frequencies but 2 context clusters
   May need more training or more data
```

**Meaning:**
- Contexts separate into clusters
- But word doesn't have enough frequencies for all senses
- **Needs more training epochs or data**

### ❌ No Polysemy
```
❌ POLYSEMY NOT EMERGED
   Word has only 1 active frequency
   Contexts cluster into 2 groups, but word doesn't separate senses
```

**Meaning:**
- Contexts are separable (good)
- But word spectrum is not sparse/multi-modal enough
- **Model hasn't learned polysemy yet**

---

## Usage in Your Code

```javascript
import { PolysemyAnalyzer } from './analysis/PolysemyAnalyzer.js';

// After training your model
const analyzer = new PolysemyAnalyzer(trainer, tokenizer);

// Analyze a polysemous word
const result = analyzer.analyzeSenseSeparation(
  'bank',           // Word to analyze
  corpus,           // Training corpus
  2                 // Expected number of senses
);

if (result) {
  console.log(`Status: ${result.verdict.status}`);
  console.log(`Active frequencies: ${result.wordSpectrum.frequencies.length}`);
  console.log(`Alignment spread: ${result.alignment.scores}`);
}
```

---

## Configuration

### Number of Clusters

```javascript
// Analyze with 2 senses (default)
analyzer.analyzeSenseSeparation('bank', corpus, 2);

// Analyze with 3 senses
analyzer.analyzeSenseSeparation('play', corpus, 3);
// Might have: play (game), play (music), play (drama)
```

### K-Means Parameters

In `PolysemyAnalyzer.js`:

```javascript
clusterContexts(spectra, k, maxIters = 20)
```

- `k`: Number of clusters (expected senses)
- `maxIters`: Maximum k-means iterations (default: 20)

---

## When to Use

### ✅ Use for:
- **Checkpoint evaluation** - Check if polysemy is emerging during training
- **Model debugging** - Verify sense separation
- **Research analysis** - Measure polysemy quality
- **Corpus evaluation** - Check if corpus has sense diversity

### ❌ Don't use for:
- **Production word sense disambiguation** - This is an analysis tool, not inference
- **Small corpora** - Need enough examples of each sense
- **Words with 1 sense** - Only analyze known polysemous words

---

## Recommended Polysemous Words to Test

| Word | Senses | Example Contexts |
|------|--------|------------------|
| bank | 2 | financial institution, river edge |
| play | 3 | game, music, drama |
| light | 2 | brightness, weight |
| right | 2 | direction, correctness |
| bat | 2 | animal, sports equipment |
| date | 2 | calendar, romantic meeting |
| club | 2 | organization, weapon |
| palm | 2 | tree, hand part |

---

## Tips

### Getting Good Results

1. **Enough examples** - Need at least 10+ occurrences per sense
2. **Distinct contexts** - Senses should appear in clearly different contexts
3. **Enough training** - Run enough epochs for frequencies to separate
4. **Proper sparsity** - If too sparse (1 freq), increase `maxFrequencies`

### Troubleshooting

**"Too few occurrences"**
- Word doesn't appear enough in corpus
- Add more data or use a different word

**"Weak sense separation"**
- Training converged but senses didn't separate
- Try more epochs or higher learning rate

**"All contexts in one cluster"**
- Corpus doesn't have diverse senses
- Add more varied contexts for the word

---

## Algorithm Details

### Context Spectrum Formula

```
S_context = (Σ S_word_i) / |context|
where word_i ∈ context, word_i ≠ target
```

### Alignment Score

Cosine similarity between word spectrum and cluster centroid:

```
alignment = (word · centroid) / (||word|| · ||centroid||)
```

### Verdict Logic

```
if numActiveFreqs == 1:
    → NO_POLYSEMY
elif numActiveFreqs < numClusters:
    → PARTIAL
elif alignmentSpread < 0.1:
    → WEAK
else:
    → SUCCESS
```

---

## Integration with Training

You can analyze polysemy at checkpoints during training:

```javascript
// In your training loop
if (epoch % 5 === 0) {
  console.log(`\n--- Polysemy Check at Epoch ${epoch} ---`);

  const analyzer = new PolysemyAnalyzer(trainer, tokenizer);
  const result = analyzer.analyzeSenseSeparation('bank', corpus, 2);

  if (result && result.verdict.status === 'SUCCESS') {
    console.log('✅ Polysemy emerged! Can stop training early.');
  }
}
```

---

## Future Enhancements

Potential additions:
- [ ] Automatic polysemous word detection
- [ ] Sense-aware similarity queries
- [ ] Visualization of cluster centroids
- [ ] Support for more than 2 senses
- [ ] Hierarchical clustering
- [ ] Context-conditioned word vectors

---

## Summary

The PolysemyAnalyzer provides **empirical verification** that CSS is learning multiple senses:

1. ✅ **Detects** if polysemy has emerged
2. ✅ **Measures** quality of sense separation
3. ✅ **Interprets** which frequencies encode which senses
4. ✅ **Validates** the core CSS hypothesis

Use it to track polysemy emergence during training and verify your model is working as intended!
