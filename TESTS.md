# Polysemy Detection Tests

This document describes the comprehensive test suite for verifying that the CSS model successfully learns polysemy (multiple word senses).

## Overview

The polysemy analysis script (`src/analyze_polysemy.js`) implements five independent tests that together provide strong evidence of whether the model has learned to separate different senses of polysemous words.

**Polysemous word example**: "bank"
- Sense 1: Financial institution (bank, loan, credit, deposit)
- Sense 2: River edge (river, shore, water, stream)

**Monosemous word example**: "oxygen"
- Single sense: Chemical element (gas, element, air, breathing)

## Running the Tests

### Analyze a Single Snapshot
```bash
node src/analyze_polysemy.js <snapshot_file> [target_word] [comparison_word]

# Examples:
node src/analyze_polysemy.js ./data/snapshots/snapshot_0001_docs5000.json bank oxygen
node src/analyze_polysemy.js ./data/snapshots/snapshot_0002_docs10000.json river ocean
```

### Analyze Evolution Across All Snapshots
```bash
node src/analyze_polysemy.js --all [target_word] [comparison_word]

# Examples:
node src/analyze_polysemy.js --all bank oxygen
node src/analyze_polysemy.js --all plant stone
```

## The Five Tests

---

## Test 1: Multiple Dominant Peaks

### What It Tests
A polysemous word should have **multiple high-amplitude frequencies** in its spectrum, each corresponding to a different sense. A monosemous word should have **one dominant frequency**.

### How It Works
1. Extract the word's spectrum (frequencies and amplitudes)
2. Find all peaks above threshold (default: amplitude > 0.3)
3. Count the number of dominant peaks
4. Compare polysemous vs. monosemous words

### Example Output
```
BANK:
  Dominant peaks (amplitude > 0.3):
    1. Freq 12: amplitude 0.4201   ← finance sense
    2. Freq 87: amplitude 0.3892   ← money sense
    3. Freq 203: amplitude 0.4156  ← river sense
  Total peaks: 3
  Potentially polysemous: YES ✓

OXYGEN (comparison):
  Dominant peaks (amplitude > 0.3):
    1. Freq 55: amplitude 0.9342
  Total peaks: 1
  Potentially polysemous: NO ✓
```

### Interpretation
- **≥2 peaks**: Potentially polysemous (PASS)
- **1 peak**: Likely monosemous (FAIL for polysemy test)

### Why This Matters
If the model learned polysemy, different contexts should activate different frequency combinations. Multiple strong peaks indicate the word participates in multiple distinct semantic spaces.

---

## Test 2: Divergence Over Training Snapshots

### What It Tests
During training, a polysemous word's multiple peaks should **grow at different rates** as the model encounters different contexts. A monosemous word should consistently strengthen **one dominant peak**.

### How It Works
1. Load all training snapshots (e.g., at 5k, 10k, 15k documents)
2. Track each frequency's amplitude over time
3. Compare temporal evolution for polysemous vs. monosemous words

### Example Output
```
BANK frequency evolution:
  Snapshot | Docs    | Active Freqs | Top 3 Amplitudes
  --------------------------------------------------------
         1 |    5000 |           12 | 0.124, 0.089, 0.078
         2 |   10000 |           14 | 0.287, 0.231, 0.189
         3 |   15000 |           16 | 0.420, 0.389, 0.416
         4 |   20000 |           13 | 0.441, 0.402, 0.428

OXYGEN frequency evolution:
  Snapshot | Docs    | Active Freqs | Top 3 Amplitudes
  --------------------------------------------------------
         1 |    5000 |           15 | 0.156, 0.012, 0.008
         2 |   10000 |           12 | 0.421, 0.023, 0.019
         3 |   15000 |           10 | 0.734, 0.031, 0.027
         4 |   20000 |            8 | 0.934, 0.018, 0.012
```

### Interpretation
**Polysemous word (bank)**:
- Multiple peaks grow simultaneously
- Top 3 amplitudes remain comparable (divergence)
- Active frequencies stabilize around maxFrequencies

**Monosemous word (oxygen)**:
- One peak dominates and grows much stronger
- Secondary peaks remain weak or shrink
- Active frequencies decrease (pruning weak frequencies)

### Why This Matters
This shows the **temporal signature** of polysemy emerging. If different senses appear in different parts of the corpus, you'll see different peaks rising at different moments.

---

## Test 3: Context Clustering

### What It Tests
The contexts where a word appears should naturally **cluster into groups** corresponding to different senses. Good clustering (measured by silhouette score) indicates distinct usage patterns.

### How It Works
1. Extract all contexts where the target word appears
2. Build a context spectrum for each occurrence (average of surrounding words)
3. Cluster contexts using K-means (k=2 for binary polysemy)
4. Calculate silhouette score (cluster quality metric)
5. Show example contexts from each cluster

### Example Output
```
Cluster 0: 45 contexts
Cluster 1: 38 contexts
Silhouette score: 0.4521
  (>0.3 = good separation, 0.0-0.3 = weak, <0.0 = poor)

Example contexts from each cluster:

Cluster 0 examples:
  1. the bank lends money to businesses
  2. he deposits money in the bank account
  3. the bank offers loans with low interest
  4. customers visit the bank to withdraw cash
  5. the bank approved his loan application

Cluster 1 examples:
  1. the river bank is covered with grass
  2. we sat on the bank and watched water
  3. the bank of the river is eroding
  4. fish swim near the bank where plants grow
  5. the boat is moored at the river bank
```

### Interpretation
- **Silhouette > 0.3**: Good separation, distinct senses (PASS)
- **Silhouette 0.0-0.3**: Weak separation, ambiguous (WEAK)
- **Silhouette < 0.0**: No separation, monosemous (FAIL)

### Why This Matters
If the model's embeddings capture semantic meaning, contexts with similar meanings should cluster together. Clean clustering validates that the learned representations distinguish between senses.

---

## Test 4: Mutual Reinforcement

### What It Tests
If two senses are truly distinct, they should use **different frequencies**. The frequencies dominant in one sense's contexts should NOT appear strongly in the other sense's contexts.

### How It Works
1. For each context cluster (from Test 3), identify top frequencies in cluster centroid
2. Calculate frequency overlap between clusters
3. Check if clusters use distinct frequency sets

### Example Output
```
Top frequencies for each cluster centroid:

Cluster 0 (finance):
  1. Freq 12: amplitude 0.5234
  2. Freq 87: amplitude 0.4912
  3. Freq 143: amplitude 0.3821
  4. Freq 56: amplitude 0.2891
  5. Freq 201: amplitude 0.2456

Cluster 1 (river):
  1. Freq 203: amplitude 0.5891
  2. Freq 78: amplitude 0.4234
  3. Freq 189: amplitude 0.3567
  4. Freq 12: amplitude 0.2123    ← shared with Cluster 0
  5. Freq 234: amplitude 0.1987

Frequency overlap between clusters: 18.2%
Overlapping frequencies: [12]
Clean separation: YES ✓
```

### Interpretation
- **Overlap < 30%**: Clean separation, distinct sense frequencies (PASS)
- **Overlap ≥ 30%**: High overlap, senses share semantic components (FAIL)

### Why This Matters
This is the **smoking gun** for polysemy. If finance contexts activate {12, 87, 143} and river contexts activate {203, 78, 189}, with minimal overlap, the model has learned that these senses are fundamentally different.

---

## Test 5: Spectrum Stability Under Context Substitution

### What It Tests
The word's spectrum should **resonate differently** with different context types. Similarity between the word and its contexts should be **higher for contexts matching its active senses**.

### How It Works
1. For each context cluster, compute average similarity between word spectrum and context spectra
2. Compare similarities across clusters
3. Check if similarities are significantly different (>20% relative difference)

### Example Output
```
Average word-to-context similarity by cluster:

Cluster 0: 0.4521 (45 contexts)
Cluster 1: 0.2891 (38 contexts)

Differentiated by context: YES ✓
```

### Interpretation
- **Relative difference > 20%**: Context-dependent activation (PASS)
- **Relative difference ≤ 20%**: No differentiation (FAIL)

### Why This Matters
This tests the **functional consequence** of polysemy. If "bank" has multiple senses, its spectrum should align better with finance contexts when used in finance, and better with geography contexts when used in geography. This validates that the learned representation is **contextually sensitive**.

---

## Final Scoring

### Polysemy Score Calculation

Each test contributes to the final score:
- Test 1: 1 point if ≥2 peaks, 0 otherwise
- Test 2: Qualitative (visual inspection of divergence)
- Test 3: 1 point if silhouette >0.3, 0.5 if >0.0, 0 otherwise
- Test 4: 1 point if overlap <30%, 0 otherwise
- Test 5: 1 point if differentiated, 0 otherwise

### Example Final Report
```
======================================================================
POLYSEMY DETECTION SUMMARY
======================================================================

Word: "bank"

Test Results:
  ✓ Test 1 (Multiple Peaks): PASS - 3 dominant peaks
  ? Test 2 (Divergence): See temporal evolution above
  ✓ Test 3 (Clustering): PASS - Silhouette 0.4521
  ✓ Test 4 (Mutual Reinforcement): PASS - 18.2% overlap
  ✓ Test 5 (Context Substitution): PASS - Contexts differentiated

Polysemy Score: 4/4
Confidence: 100.0%

✓ POLYSEMY DETECTED: Model successfully learned multiple senses
```

### Confidence Levels
- **≥60%**: Strong evidence of polysemy (PASS)
- **30-60%**: Weak evidence, needs more training (WEAK)
- **<30%**: No evidence, likely monosemous (FAIL)

---

## Common Polysemous Words to Test

Good test cases for polysemy:

### High Polysemy (Multiple Distinct Senses)
- **bank**: financial institution / river edge
- **plant**: living organism / factory
- **bat**: flying mammal / sports equipment
- **spring**: season / coiled metal / water source
- **court**: legal venue / royal residence / sports area
- **date**: calendar day / romantic meeting / fruit
- **band**: musical group / elastic strip / range
- **nail**: finger/toe covering / metal fastener
- **rock**: stone / music genre / motion
- **light**: illumination / not heavy

### Moderate Polysemy (Related Senses)
- **run**: move fast / operate / campaign / flow
- **set**: put down / collection / ready / harden
- **break**: separate / pause / dawn
- **book**: publication / reserve
- **lead**: guide / metal / leash

### Expected Monosemous (Single Sense)
- **oxygen**: chemical element
- **hydrogen**: chemical element
- **nitrogen**: chemical element
- **photosynthesis**: biological process
- **algorithm**: computational procedure
- **chromosome**: genetic structure

---

## Interpreting Results

### Strong Polysemy Signal
When all 5 tests pass, you have **overwhelming evidence** that:
1. The word's spectrum has multiple distinct components (Test 1)
2. These components emerged gradually during training (Test 2)
3. Usage contexts naturally separate into groups (Test 3)
4. Each group activates different frequencies (Test 4)
5. The word responds differently to different context types (Test 5)

### Weak or No Polysemy
If tests fail:
- **Test 1 fails**: Word may need more training data or is truly monosemous
- **Test 3 fails**: Contexts don't cluster well - may need larger window size
- **Test 4 fails**: High overlap suggests senses share semantic features
- **Test 5 fails**: Word spectrum doesn't differentiate contexts

### Troubleshooting

**Problem**: All words show low scores
- **Cause**: Model undertrained or wrong hyperparameters
- **Fix**: Train longer, adjust sparsityPenalty, increase maxFrequencies

**Problem**: High-frequency words dominate results
- **Cause**: Stopwords not filtered, or poor initialization
- **Fix**: Use `frequency-scaled` initialization strategy in `prepare_vocabulary.js`

**Problem**: Context clustering always fails
- **Cause**: Window size too small, not enough context
- **Fix**: Increase windowSize in CONFIG (try 3-5)

**Problem**: Word not found in vocabulary
- **Cause**: Word didn't appear in training corpus
- **Fix**: Use more training data or lower minFrequency in vocabulary preparation

---

## Advanced Usage

### Custom Threshold for Peaks
Modify in `src/analyze_polysemy.js`:
```javascript
const CONFIG = {
    multiPeakThreshold: 0.3,  // Lower = more peaks detected
    // ...
};
```

### Analyze Three-Sense Polysemy
For words with 3+ senses (e.g., "spring"), modify the clustering:
```javascript
// In analyze_polysemy.js, change k-means k parameter:
const clustering = clusterContexts(contextSpectra, 3);  // 3 clusters instead of 2
```

### Compare Multiple Words
Run analysis in a loop:
```bash
for word in bank plant spring bat rock; do
    echo "Analyzing: $word"
    node src/analyze_polysemy.js ./data/snapshots/snapshot_0002_docs10000.json $word oxygen
done
```

---

## References

For theoretical background on polysemy and spectral semantics:
- See [docs/ABSTRACT.md](docs/ABSTRACT.md) for the CSS framework
- See [docs/TRAINING.md](docs/TRAINING.md) for training paradigm
- See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for practical training tips

## Questions?

Common questions:

**Q: How many training documents are needed to see polysemy emerge?**
A: Typically 10,000-50,000 documents for common words. Rare words need more data.

**Q: Can I test specific word pairs (antonyms, synonyms)?**
A: Yes! Use the similarity functions or modify analyze_polysemy.js to compare any words.

**Q: What if my polysemous word only shows 1 peak?**
A: It needs more training data, or increase maxFrequencies to allow more peaks.

**Q: How do I visualize the frequency spectra?**
A: You can export snapshot data and plot amplitudes vs. frequency indices in Python/matplotlib.
