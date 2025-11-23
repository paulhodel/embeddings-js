# Stability Analysis Guide

## Overview

The **StabilityAnalyzer** measures spectrum stability over training by comparing word spectra between different checkpoints. This helps detect training issues like:

- **Early training**: Big changes (model actively learning)
- **Later training**: Smaller changes (model stabilizing)
- **Issues**: Wild changes forever (unstable/high LR) or freeze too early (low LR)

---

## What It Does

For each probe word:

1. **Loads two checkpoints** (e.g., epoch 2 and epoch 4)
2. **Compares word spectra** between checkpoints
3. **Measures similarity** (cosine similarity between dense vectors)
4. **Tracks changes**:
   - Sparsity change (number of active frequencies)
   - Frequency turnover (which frequencies changed)
   - Amplitude change (for kept frequencies)
5. **Delivers verdict** on training stability

---

## Quick Start

### Basic Usage: Compare Two Checkpoints

```bash
# Compare two model snapshots
npm run stability models/snapshots/model_epoch2.json models/snapshots/model_epoch4.json

# Analyze specific words
npm run stability models/snapshots/model_epoch2.json models/snapshots/model_epoch4.json bank,river,money
```

### Time Series Analysis: Multiple Checkpoints

```bash
# Analyze stability over multiple epochs (uses glob pattern)
npm run stability "models/snapshots/model_epoch*.json"
```

---

## Example Output

### Comparing Two Checkpoints

```
======================================================================
STABILITY ANALYSIS: Comparing Checkpoints
======================================================================

Loading checkpoints...
  Checkpoint 1: models/snapshots/model_epoch2.json
  Checkpoint 2: models/snapshots/model_epoch4.json

Analyzing top 20 frequent words

Comparing spectra:
  "the" - similarity: 0.8234, sparsity: 5 → 6 (Δ1)
  "and" - similarity: 0.7892, sparsity: 4 → 5 (Δ1)
  "bank" - similarity: 0.6543, sparsity: 3 → 4 (Δ1)
  "river" - similarity: 0.7123, sparsity: 4 → 4 (Δ0)
  ...

======================================================================
STABILITY SUMMARY
======================================================================

Words analyzed: 20

Spectral similarity (0=changed completely, 1=unchanged):
  Mean:   0.7234
  Median: 0.7456
  Min:    0.5123 (word: "bank")
  Max:    0.9234 (word: "the")

Sparsity change:
  Mean:   1.23 frequencies
  Median: 1.00 frequencies

Frequency turnover:
  Mean:   25.34%
  Median: 23.50%

----------------------------------------------------------------------
VERDICT:
----------------------------------------------------------------------
✅ STABILIZING (similarity: 0.723)
   Spectra converging but still refining
   Frequency turnover: 25.3%
   This is expected in late training
   Model approaching convergence
----------------------------------------------------------------------
```

### Time Series Analysis

```
██████████████████████████████████████████████████████████████████████
TIME SERIES STABILITY ANALYSIS
██████████████████████████████████████████████████████████████████████

Found 3 checkpoints:

  1. model_epoch2.json
  2. model_epoch4.json
  3. model_epoch6.json

Comparing checkpoint 1 → 2...
[... analysis for epoch 2→4 ...]

Comparing checkpoint 2 → 3...
[... analysis for epoch 4→6 ...]

██████████████████████████████████████████████████████████████████████
TIME SERIES SUMMARY
██████████████████████████████████████████████████████████████████████

Similarity over time (higher = more stable):
Step | Mean Similarity | Turnover | Status
------------------------------------------------------------
2→4  | 0.723           | 25.3%    | STABILIZING
4→6  | 0.842           | 15.2%    | STABILIZING

----------------------------------------------------------------------
TRENDS:
----------------------------------------------------------------------
✅ Spectra are STABILIZING over time (good)
----------------------------------------------------------------------
```

---

## How It Works

### Step 1: Load Checkpoints

Loads two model checkpoints saved during training:

```javascript
const model1 = ModelPersistence.loadModel('models/snapshots/model_epoch2.json');
const model2 = ModelPersistence.loadModel('models/snapshots/model_epoch4.json');
```

Each checkpoint contains:
- Model configuration
- Vocabulary
- Word spectra (frequencies + amplitudes)
- Word frequencies

### Step 2: Select Probe Words

Either:
- **Auto-select**: Top 20 most frequent words (default)
- **Manual**: Specify words via command line

```javascript
// Auto-select frequent words
const wordFreqs = Array.from(model1.wordFreq.entries())
  .sort((a, b) => b[1] - a[1])
  .slice(0, 20)
  .map(([word]) => word);
```

### Step 3: Compare Spectra

For each word, compute:

**1. Spectral Similarity (Cosine Similarity)**

Measures overall similarity between dense vectors:

```javascript
similarity = (vec1 · vec2) / (||vec1|| * ||vec2||)
```

- **1.0**: Identical (no change)
- **0.0**: Completely different

**2. Sparsity Change**

How many active frequencies changed:

```javascript
sparsity1 = spectrum1.frequencies.length  // e.g., 5
sparsity2 = spectrum2.frequencies.length  // e.g., 6
sparsityChange = Math.abs(sparsity2 - sparsity1)  // Δ1
```

**3. Frequency Turnover**

Which specific frequencies changed:

```javascript
freqSet1 = new Set(spectrum1.frequencies)  // [2, 5, 10, 15, 20]
freqSet2 = new Set(spectrum2.frequencies)  // [2, 5, 10, 18, 22, 25]

kept = 3     // frequencies in both: 2, 5, 10
added = 3    // new frequencies: 18, 22, 25
removed = 2  // removed frequencies: 15, 20

turnoverRate = (added + removed) / totalUnique * 100
            = (3 + 2) / 8 * 100 = 62.5%
```

**4. Amplitude Change**

For frequencies that appear in both checkpoints:

```javascript
amplitudeChange = Σ |amp2 - amp1| / numKept
```

### Step 4: Aggregate Statistics

Compute statistics across all probe words:

```javascript
meanSimilarity = mean(similarities)
medianSimilarity = median(similarities)
minSimilarity = min(similarities)  // Most changed word
maxSimilarity = max(similarities)  // Most stable word
```

### Step 5: Deliver Verdict

Based on mean similarity:

```javascript
if (meanSimilarity > 0.95):
  → FROZEN (learning rate too low)
else if (meanSimilarity < 0.3):
  → UNSTABLE (learning rate too high)
else if (meanSimilarity < 0.6):
  → LEARNING (healthy active learning)
else:
  → STABILIZING (converging)
```

---

## Interpretation

### ✅ STABILIZING (0.6 - 0.95)

```
✅ STABILIZING (similarity: 0.723)
   Spectra converging but still refining
   Frequency turnover: 25.3%
   This is expected in late training
   Model approaching convergence
```

**Meaning:**
- Spectra are changing at a healthy rate
- Model is refining representations
- **Action**: Continue training, model converging normally

---

### ✅ LEARNING (< 0.6)

```
✅ ACTIVELY LEARNING (similarity: 0.534)
   Spectra changing at healthy rate
   Frequency turnover: 45.2%
   This is expected in early training
   Continue training and check stability later
```

**Meaning:**
- Model actively learning representations
- Large changes expected in early epochs
- **Action**: Continue training, check stability later

---

### ⚠️ FROZEN (> 0.95)

```
⚠️  SPECTRA FROZEN (similarity: 0.972)
   Spectra barely changing between checkpoints
   Possible causes:
   - Learning rate too low
   - Sparsity penalty too aggressive
   - Already converged (check loss)
   Recommendation: Increase learning rate or reduce sparsity penalty
```

**Meaning:**
- Spectra not changing enough between epochs
- Model may not be learning effectively
- **Action**:
  - Increase learning rate (e.g., 0.01 → 0.03)
  - Reduce sparsity penalty (e.g., 0.005 → 0.002)
  - Check if loss is already converged

---

### ⚠️ UNSTABLE (< 0.3)

```
⚠️  UNSTABLE TRAINING (similarity: 0.234)
   Spectra changing wildly between checkpoints
   Possible causes:
   - Learning rate too high
   - Training not converging
   - Model oscillating
   Recommendation: Reduce learning rate or add momentum
```

**Meaning:**
- Spectra changing too dramatically
- Training may be oscillating or diverging
- **Action**:
  - Reduce learning rate (e.g., 0.1 → 0.03)
  - Add momentum or use adaptive learning rate
  - Check for NaN values in loss

---

## Configuration

### Training Configuration (train.js)

Control snapshot saving frequency:

```javascript
const CONFIG = {
  // Model Snapshots (for stability analysis)
  saveSnapshotEvery: 2,         // Save model snapshot every N epochs
  snapshotDir: './models/snapshots',

  // Training
  epochs: 10,
  learningRate: 0.03,
  sparsityPenalty: 0.003,
  // ...
};
```

### Snapshot Behavior

During training, snapshots are saved automatically:

```
Epoch 1/10:
  Loss: 0.234567
  Avg sparsity: 4.23 frequencies/word

Epoch 2/10:
  Loss: 0.198765
  Avg sparsity: 4.56 frequencies/word
  💾 Saving snapshot: ./models/snapshots/model_epoch2.json

Epoch 3/10:
  Loss: 0.176543
  ...
```

Snapshots include:
- Full model state (spectra, vocabulary)
- Configuration
- Training statistics
- Timestamp

---

## Usage in Your Code

```javascript
import { StabilityAnalyzer } from './analysis/StabilityAnalyzer.js';

// Compare two checkpoints
const result = StabilityAnalyzer.compareCheckpoints(
  'models/snapshots/model_epoch2.json',
  'models/snapshots/model_epoch4.json',
  ['bank', 'river', 'money']  // Optional: specific words
);

console.log(`Status: ${result.verdict.status}`);
console.log(`Mean similarity: ${result.stats.meanSimilarity}`);

// Time series analysis
const timeSeries = StabilityAnalyzer.analyzeStabilityTimeSeries(
  [
    'models/snapshots/model_epoch2.json',
    'models/snapshots/model_epoch4.json',
    'models/snapshots/model_epoch6.json'
  ],
  null  // Auto-select frequent words
);

// Check for trends
const similarities = timeSeries.map(r => r.stats.meanSimilarity);
const early = similarities[0];
const late = similarities[similarities.length - 1];

if (late > early + 0.2) {
  console.log('✅ Model is stabilizing over time');
}
```

---

## When to Use

### ✅ Use For:

- **Checkpoint evaluation** - Track training progress
- **Hyperparameter tuning** - Detect LR issues early
- **Model debugging** - Identify convergence problems
- **Research analysis** - Measure training dynamics

### ❌ Don't Use For:

- **Real-time training monitoring** - This loads full checkpoints (slow)
- **Single checkpoint analysis** - Need at least 2 checkpoints to compare
- **Small models** - More useful for large-scale training

---

## Command Line Examples

### Compare Two Specific Checkpoints

```bash
npm run stability models/snapshots/model_epoch2.json models/snapshots/model_epoch4.json
```

### Analyze Specific Words

```bash
npm run stability models/snapshots/model_epoch2.json models/snapshots/model_epoch4.json bank,river,money,loan
```

### Time Series Analysis (All Snapshots)

```bash
npm run stability "models/snapshots/model_epoch*.json"
```

### Time Series with Specific Words

```bash
npm run stability "models/snapshots/model_epoch*.json" bank,river
```

---

## Integration with Training

Stability analysis is automatically enabled in training:

```javascript
// In train.js
const CONFIG = {
  saveSnapshotEvery: 2,  // Save every 2 epochs
  snapshotDir: './models/snapshots',
  epochs: 10
};

// Snapshots saved at: epoch 2, 4, 6, 8, 10
// Final model saved separately at: ./models/css_model.json
```

After training completes, analyze stability:

```bash
# Compare early vs mid training
npm run stability models/snapshots/model_epoch2.json models/snapshots/model_epoch6.json

# Full time series
npm run stability "models/snapshots/model_epoch*.json"
```

---

## Troubleshooting

### Issue: "Checkpoint not found"

**Cause**: File path incorrect or snapshots not saved

**Solution**:
1. Check that training config has `saveSnapshotEvery > 0`
2. Verify `snapshotDir` exists after training
3. Use absolute paths or paths relative to project root

### Issue: "Word not in both checkpoints"

**Cause**: Vocabulary changed between checkpoints

**Solution**:
- Only compare checkpoints from same training run
- Don't compare models trained on different corpora
- Use auto-select mode (default) instead of manual words

### Issue: "Similarity always near 1.0"

**Cause**: Learning rate too low or checkpoints too close

**Solution**:
- Compare checkpoints further apart (e.g., epoch 2 vs 10, not 2 vs 3)
- Increase learning rate in training config
- Reduce sparsity penalty

### Issue: "Similarity very low (< 0.3)"

**Cause**: Learning rate too high or model unstable

**Solution**:
- Reduce learning rate (e.g., 0.1 → 0.03)
- Check for NaN in loss values
- Try smaller batch size or more epochs

---

## Algorithm Details

### Cosine Similarity Formula

```
similarity(v1, v2) = (v1 · v2) / (||v1|| × ||v2||)

where:
  v1 · v2 = Σ(v1[i] × v2[i])
  ||v|| = √(Σ(v[i]²))
```

### Frequency Turnover Formula

```
freqSet1 = {ω | ω ∈ spectrum1}
freqSet2 = {ω | ω ∈ spectrum2}

kept = |freqSet1 ∩ freqSet2|
added = |freqSet2 \ freqSet1|
removed = |freqSet1 \ freqSet2|

turnover = (added + removed) / |freqSet1 ∪ freqSet2| × 100%
```

### Verdict Thresholds

```
if meanSimilarity > 0.95:
    verdict = FROZEN
elif meanSimilarity < 0.3:
    verdict = UNSTABLE
elif meanSimilarity < 0.6:
    verdict = LEARNING
else:
    verdict = STABILIZING
```

---

## Tips

### Getting Good Results

1. **Compare appropriate epochs**
   - Early: Compare epoch 1 vs 2 (expect low similarity)
   - Late: Compare epoch 8 vs 10 (expect high similarity)

2. **Use time series for trends**
   - Single comparison shows snapshot
   - Time series shows trajectory

3. **Check word diversity**
   - Use auto-select (gets frequent words)
   - Or manually select diverse words (common + rare)

4. **Save enough snapshots**
   - `saveSnapshotEvery: 1` → All epochs (slow, large storage)
   - `saveSnapshotEvery: 2` → Every other epoch (recommended)
   - `saveSnapshotEvery: 5` → Every 5 epochs (coarse-grained)

---

## Summary

The StabilityAnalyzer provides **empirical verification** of training dynamics:

1. ✅ **Detects** training issues (frozen/unstable)
2. ✅ **Measures** convergence rate
3. ✅ **Tracks** spectrum evolution over time
4. ✅ **Guides** hyperparameter tuning

Use it to monitor training health and ensure your model is learning effectively!
