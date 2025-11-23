# Debug Utilities Guide

## Overview

The **DebugUtils** module provides fast, interactive tools for qualitative model inspection. These are designed for quick sanity checks during and after training to verify that:

- ✅ **Meaningful stable peaks** are appearing in word spectra
- ✅ **Noise peaks** are disappearing over time
- ✅ **Spectrum shapes** stop jittering (converging)
- ✅ **Frequencies are interpretable** (semantic factors)

---

## Quick Start

### Basic Commands

```bash
# Show a word's spectrum
npm run debug models/css_model.json word bank

# Show which words use a frequency (reverse lookup)
npm run debug models/css_model.json freq 37

# Compare multiple word spectra
npm run debug models/css_model.json compare bank,river,money

# Show most interpretable frequencies
npm run debug models/css_model.json interpretable

# Show sparsity statistics
npm run debug models/css_model.json sparsity
```

---

## Commands Reference

### 1. `word` - Show Word Spectrum

**Usage:**
```bash
npm run debug <model_path> word <word> [topK]
```

**Example:**
```bash
npm run debug models/css_model.json word bank
```

**Output:**
```
======================================================================
WORD SPECTRUM
======================================================================

"bank":
  (freq   37) amp=0.8123
  (freq  412) amp=0.7456
  (freq   89) amp=0.2234
  (freq  156) amp=0.1876
  (freq   23) amp=0.0987
```

**Interpretation:**
- **High amplitude** (>0.5): Dominant semantic factors
- **Medium amplitude** (0.1-0.5): Secondary meanings
- **Low amplitude** (<0.1): Weak associations or noise

**Run at different training steps to see:**
- New meaningful peaks appearing (model learning)
- Noise peaks disappearing (model converging)
- Spectrum stabilizing (amplitudes stop changing)

---

### 2. `freq` - Frequency Reverse Lookup

**Usage:**
```bash
npm run debug <model_path> freq <freqIndex> [topK]
```

**Example:**
```bash
npm run debug models/css_model.json freq 37 10
```

**Output:**
```
======================================================================
FREQUENCY REVERSE LOOKUP
======================================================================

Frequency 37:
  bank                 (0.8123)
  money                (0.7845)
  loan                 (0.7123)
  account              (0.6543)
  credit               (0.6234)
  financial            (0.5987)
  deposit              (0.5432)
  interest             (0.5123)
  mortgage             (0.4987)
  savings              (0.4765)

  ... 45 total words use this frequency
```

**This is gold!** 🏆

If you see coherent semantic groups, it means:
- ✅ Frequency 37 encodes "financial/banking" concept
- ✅ Model is learning interpretable factors
- ✅ CSS hypothesis is working

**What to look for:**
- **Good**: Semantically related words clustered together
- **Bad**: Random unrelated words (frequency is noise)

**Multiple interpretable frequencies = polysemy:**

```bash
npm run debug models/css_model.json freq 412
```

```
Frequency 412:
  bank                 (0.7456)
  river                (0.6987)
  stream               (0.6321)
  shore                (0.5876)
  water                (0.5432)
  ...
```

Now frequency 412 encodes "river/nature" concept!

This proves "bank" has learned multiple senses (financial + river).

---

### 3. `compare` - Compare Word Spectra

**Usage:**
```bash
npm run debug <model_path> compare <word1,word2,...> [topK]
```

**Example:**
```bash
npm run debug models/css_model.json compare bank,river,money
```

**Output:**
```
======================================================================
WORD SPECTRA COMPARISON
======================================================================

"bank":
  (freq   37) amp=0.8123
  (freq  412) amp=0.7456
  (freq   89) amp=0.2234
  (freq  156) amp=0.1876
  (freq   23) amp=0.0987

"river":
  (freq  412) amp=0.6987
  (freq   89) amp=0.5432
  (freq  234) amp=0.4321
  (freq  156) amp=0.3876
  (freq   67) amp=0.2123

"money":
  (freq   37) amp=0.7845
  (freq   23) amp=0.6543
  (freq  156) amp=0.5234
  (freq   89) amp=0.4321
  (freq  178) amp=0.3456
```

**Observations:**
- **bank** and **river** share freq 412 (river sense)
- **bank** and **money** share freq 37 (financial sense)
- **bank** has both → polysemy detected! ✅

**Use cases:**
- Compare synonyms (should share frequencies)
- Compare antonyms (should have different frequencies)
- Verify polysemy (shared frequencies with different senses)

---

### 4. `freqs` - Compare Multiple Frequencies

**Usage:**
```bash
npm run debug <model_path> freqs <freq1,freq2,...> [topWords]
```

**Example:**
```bash
npm run debug models/css_model.json freqs 37,412,89
```

**Output:**
```
======================================================================
FREQUENCY COMPARISON - Top Words per Frequency
======================================================================

Frequency 37:
  bank                 (0.8123)
  money                (0.7845)
  loan                 (0.7123)
  account              (0.6543)
  credit               (0.6234)
  ...

Frequency 412:
  bank                 (0.7456)
  river                (0.6987)
  stream               (0.6321)
  shore                (0.5876)
  water                (0.5432)
  ...

Frequency 89:
  flow                 (0.8234)
  current              (0.7654)
  stream               (0.7123)
  river                (0.5432)
  bank                 (0.2234)
  ...
```

**Interpretation:**
- Freq 37: Financial cluster
- Freq 412: River/geographic cluster
- Freq 89: Flow/movement cluster

---

### 5. `interpretable` - Most Interpretable Frequencies

**Usage:**
```bash
npm run debug <model_path> interpretable [topN] [topWords]
```

**Example:**
```bash
npm run debug models/css_model.json interpretable 5 8
```

**Output:**
```
██████████████████████████████████████████████████████████████████████
MOST INTERPRETABLE FREQUENCIES
██████████████████████████████████████████████████████████████████████

Frequencies with highest variance = clearest semantic factors

1. Frequency 37
   Words using it: 45
   Variance: 0.1234 (higher = more interpretable)
   Max amplitude: 0.8123

   Top words:
     bank                 (0.8123)
     money                (0.7845)
     loan                 (0.7123)
     account              (0.6543)
     credit               (0.6234)
     financial            (0.5987)
     deposit              (0.5432)
     interest             (0.5123)

  ... 45 total words use this frequency

2. Frequency 412
   Words using it: 38
   Variance: 0.1098
   Max amplitude: 0.7456

   Top words:
     river                (0.6987)
     stream               (0.6321)
     bank                 (0.7456)
     shore                (0.5876)
     water                (0.5432)
     lake                 (0.5234)
     tributary            (0.5012)
     creek                (0.4876)

  ... 38 total words use this frequency

...
```

**What is "interpretability"?**

A frequency is **interpretable** if:
- It's used by a coherent semantic cluster
- High variance in amplitudes (clear separation)
- Not used by random unrelated words

**High variance = interpretable:**
- Some words have very high amplitude (core members)
- Some words have low amplitude (peripheral members)
- Not all words at same amplitude (not noise)

**Use this to:**
- ✅ Verify model is learning semantic factors
- ✅ Identify the clearest "meaning dimensions"
- ✅ Debug training (if no interpretable frequencies → problem)

---

### 6. `sparsity` - Sparsity Statistics

**Usage:**
```bash
npm run debug <model_path> sparsity
```

**Output:**
```
======================================================================
SPARSITY STATISTICS
======================================================================

Vocabulary size: 5000
Total active frequencies: 23450

Active frequencies per word:
  Mean:   4.69
  Median: 5
  Min:    1
  Max:    8

Distribution:
   1 freqs:   234 words ( 4.7%) ███
   2 freqs:   456 words ( 9.1%) ████
   3 freqs:   789 words (15.8%) ███████
   4 freqs:   945 words (18.9%) █████████
   5 freqs:  1234 words (24.7%) ████████████
   6 freqs:   876 words (17.5%) ████████
   7 freqs:   345 words ( 6.9%) ███
   8 freqs:   121 words ( 2.4%) █

======================================================================
```

**Interpretation:**

**Good signs:**
- Bell curve distribution (most words have moderate sparsity)
- Mean around 4-6 frequencies
- Few words with 1 frequency (monosemous)
- Few words with max frequencies (polysemous)

**Bad signs:**
- All words have 1 frequency (model not learning complexity)
- All words have max frequencies (not sparse, too dense)
- Uniform distribution (no structure)

**Sparsity levels:**
- 1-2 frequencies: Simple, monosemous words
- 3-5 frequencies: Normal words with some complexity
- 6-8 frequencies: Complex, polysemous words
- >8 frequencies: Very polysemous or noisy

---

### 7. `extreme` - Extreme Sparsity Words

**Usage:**
```bash
npm run debug <model_path> extreme [topK]
```

**Example:**
```bash
npm run debug models/css_model.json extreme 10
```

**Output:**
```
======================================================================
EXTREME SPARSITY WORDS
======================================================================

MOST active frequencies (complex/polysemous words):
   1. bank                  (8 frequencies)
   2. set                   (8 frequencies)
   3. run                   (8 frequencies)
   4. right                 (7 frequencies)
   5. play                  (7 frequencies)
   6. light                 (7 frequencies)
   7. left                  (7 frequencies)
   8. bear                  (7 frequencies)
   9. round                 (6 frequencies)
  10. head                  (6 frequencies)

LEAST active frequencies (simple/monosemous words):
   1. aardvark              (1 frequency)
   2. zeitgeist             (1 frequency)
   3. quintessential        (1 frequency)
   4. serendipity           (1 frequency)
   5. ubiquitous            (1 frequency)
   6. ephemeral             (1 frequency)
   7. ambiguous             (1 frequency)
   8. dichotomy             (1 frequency)
   9. esoteric              (1 frequency)
  10. juxtaposition         (1 frequency)

======================================================================
```

**Observations:**

**Most complex (8 freqs):**
- Known polysemous words: bank, set, run, right, play
- These SHOULD have many frequencies (multiple meanings)
- ✅ Model correctly learned complexity

**Least complex (1 freq):**
- Rare or technical words: aardvark, zeitgeist, quintessential
- Single, specific meaning
- ✅ Model correctly learned simplicity

**This validates the CSS hypothesis!** 🎉

---

### 8. `evolve` - Track Word Evolution

**Usage:**
```bash
npm run debug <checkpoint_path> evolve <word> <checkpoint1,checkpoint2,...> [topK]
```

**Example:**
```bash
npm run debug models/snapshots/model_epoch2.json evolve bank models/snapshots/model_epoch2.json,models/snapshots/model_epoch4.json,models/snapshots/model_epoch6.json
```

**Output:**
```
======================================================================
TRACKING SPECTRUM EVOLUTION: "bank"
======================================================================

Checkpoint 1: model_epoch2.json
    (freq   37) amp=0.6234
    (freq  412) amp=0.5123
    (freq   89) amp=0.3456
    (freq  156) amp=0.2345
    (freq   23) amp=0.1234

Checkpoint 2: model_epoch4.json
    (freq   37) amp=0.7456
    (freq  412) amp=0.6789
    (freq   89) amp=0.2987
    (freq  156) amp=0.2123
    (freq   23) amp=0.1456

Checkpoint 3: model_epoch6.json
    (freq   37) amp=0.8123
    (freq  412) amp=0.7456
    (freq   89) amp=0.2234
    (freq  156) amp=0.1876
    (freq   23) amp=0.0987

======================================================================
Look for:
  ✓ New meaningful stable peaks appearing
  ✓ Noise peaks disappearing
  ✓ Spectrum shape stops jittering
======================================================================
```

**Analysis:**

**Checkpoint 1 → 2:**
- Freq 37 amp: 0.623 → 0.746 (+12%) ✓ Growing (learning)
- Freq 412 amp: 0.512 → 0.679 (+17%) ✓ Growing (learning)
- Freq 89 amp: 0.346 → 0.299 (-5%) → Shrinking (less important)

**Checkpoint 2 → 3:**
- Freq 37 amp: 0.746 → 0.812 (+7%) ✓ Still growing but slower (stabilizing)
- Freq 412 amp: 0.679 → 0.746 (+7%) ✓ Still growing but slower (stabilizing)
- Freq 23 amp: 0.146 → 0.099 (-33%) → Disappearing (noise)

**Conclusions:**
- ✅ Frequencies 37 and 412 are stable meaningful peaks (financial + river)
- ✅ Frequency 23 is disappearing (was noise)
- ✅ Model is converging (amplitude changes slowing down)

---

## Typical Workflow

### During Training

**Every few epochs:**

```bash
# Check a polysemous word
npm run debug models/snapshots/model_epoch2.json word bank

# Check if frequencies are interpretable
npm run debug models/snapshots/model_epoch2.json freq 37
npm run debug models/snapshots/model_epoch2.json freq 412

# Check sparsity distribution
npm run debug models/snapshots/model_epoch2.json sparsity
```

**Look for:**
- Amplitudes increasing for meaningful frequencies
- Noise frequencies disappearing
- Sparsity distribution normalizing

### After Training

**Full model analysis:**

```bash
# Find most interpretable frequencies
npm run debug models/css_model.json interpretable 10

# Check if polysemy emerged
npm run debug models/css_model.json compare bank,river,money

# Verify sparsity is reasonable
npm run debug models/css_model.json sparsity

# Check extreme cases
npm run debug models/css_model.json extreme 20
```

### Debugging Issues

**Issue: Random word spectra**

```bash
# Check if any frequencies are interpretable
npm run debug models/css_model.json interpretable 20

# If all frequencies have random words → training failed
```

**Issue: All words have 1 frequency**

```bash
npm run debug models/css_model.json sparsity

# Check distribution
# If mean=1, median=1 → sparsity penalty too high
```

**Issue: Polysemy not emerging**

```bash
# Check known polysemous words
npm run debug models/css_model.json word bank
npm run debug models/css_model.json word set
npm run debug models/css_model.json word run

# If all have 1-2 frequencies → need more epochs or data
```

---

## Integration with Code

```javascript
import { DebugUtils } from './utils/DebugUtils.js';
import { ModelPersistence } from './utils/ModelPersistence.js';
import { CSSTrainer } from './core/CSSTrainer.js';
import { Tokenizer } from './preprocessing/tokenizer.js';

// Load model
const model = ModelPersistence.loadModel('models/css_model.json');
const trainer = new CSSTrainer(model.config);
trainer.initialize(model.modelData.vocabSize);
trainer.importModel(model.modelData);

const tokenizer = new Tokenizer();
tokenizer.vocab = new Map(Object.entries(model.vocab));
tokenizer.wordFreq = new Map(Object.entries(model.wordFreq));
// ... restore tokenizer state

// Print word spectrum
DebugUtils.printWordSpectrum('bank', trainer, tokenizer);

// Show frequency reverse lookup
DebugUtils.showTopWordsForFrequency(37, trainer, tokenizer, 10);

// Compare words
DebugUtils.printWordSpectraComparison(['bank', 'river', 'money'], trainer, tokenizer);

// Find interpretable frequencies
const interpretable = DebugUtils.findMostInterpretableFrequencies(trainer, tokenizer, 10);
console.log(interpretable);

// Show most interpretable
DebugUtils.showMostInterpretableFrequencies(trainer, tokenizer, 5, 8);

// Sparsity stats
DebugUtils.printSparsityStats(trainer, tokenizer);

// Extreme sparsity
DebugUtils.showExtremeSparsity(trainer, tokenizer, 10);

// Track evolution
DebugUtils.trackWordEvolution('bank', [
  'models/snapshots/model_epoch2.json',
  'models/snapshots/model_epoch4.json',
  'models/snapshots/model_epoch6.json'
], 5);
```

---

## Command Summary

| Command | Purpose | Example |
|---------|---------|---------|
| `word` | Show word's spectrum | `npm run debug model.json word bank` |
| `freq` | Reverse lookup (words using frequency) | `npm run debug model.json freq 37` |
| `compare` | Compare word spectra | `npm run debug model.json compare bank,river` |
| `freqs` | Compare frequencies | `npm run debug model.json freqs 37,412` |
| `interpretable` | Most interpretable frequencies | `npm run debug model.json interpretable` |
| `sparsity` | Sparsity distribution | `npm run debug model.json sparsity` |
| `extreme` | Extreme sparsity words | `npm run debug model.json extreme` |
| `evolve` | Track word evolution | `npm run debug model.json evolve bank paths...` |

---

## Tips

### Finding Good Probe Words

**Common polysemous words to test:**
- bank (financial, river)
- set (group, arrange, TV, tennis)
- run (move, operate, flow)
- right (direction, correct)
- play (game, music, drama)
- light (brightness, weight)
- left (direction, past tense)
- bear (animal, carry)
- date (calendar, romantic)
- bat (animal, sports)

### Interpreting Frequency Clusters

**Good frequency (interpretable):**
```
Frequency 37:
  bank      (0.81)
  money     (0.78)
  loan      (0.71)
  account   (0.65)
  credit    (0.62)
```
→ Clear semantic cluster (financial)

**Noise frequency (not interpretable):**
```
Frequency 89:
  the       (0.34)
  bank      (0.22)
  purple    (0.19)
  running   (0.15)
  computer  (0.12)
```
→ Random words, no coherent meaning

### When to Use Each Command

- **`word`**: Check specific word you care about
- **`freq`**: Verify frequency is interpretable
- **`compare`**: Verify synonym/antonym relationships
- **`interpretable`**: Find best semantic factors
- **`sparsity`**: Check overall model health
- **`extreme`**: Validate polysemy detection
- **`evolve`**: Track convergence during training

---

## Summary

The DebugUtils provide **fast qualitative feedback** on model quality:

1. ✅ **Verify interpretability** - Frequencies encode semantic factors
2. ✅ **Detect polysemy** - Words have multiple meaningful frequencies
3. ✅ **Track convergence** - Spectra stabilize over epochs
4. ✅ **Identify issues** - Noise, overfitting, underfitting

Use these tools throughout training to ensure your CSS model is learning meaningful, sparse, interpretable representations!
