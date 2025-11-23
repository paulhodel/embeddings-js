# Initialization and Pruning Strategy

## Overview

The CSS model uses a sophisticated **dense initialization + two-phase pruning** strategy to discover optimal sparsity patterns. This approach is inspired by:

- **Neural nets** (dense random initialization)
- **Sparse coding** (dense code + later sparsification)
- **k-SVD** (dense initialization + iterative hard-thresholding)

---

## Key Principles

### 1. Dense Initialization with Tiny Amplitudes

**Strategy:**
- ALL words start with `maxFrequencies` random frequencies
- Amplitudes are TINY: [0.0005, 0.003]

**Why Dense?**
- Allows each word to explore many semantic directions
- Model discovers which frequencies are meaningful
- Polysemous words keep multiple frequencies
- Simple words prune down to 1-2 frequencies

**Why Tiny Amplitudes?**
- **CRITICAL**: Prevents early random context signals from dominating learning
- If amplitudes are large (e.g., 0.1-1.0), early random gradients cause chaos
- Tiny amplitudes → only reinforced frequencies survive
- Acts like "weak initialization" in neural nets

### 2. Two-Phase Pruning Schedule

**Phase 1 (0-20% of training): EXPLORATION**
- **Purpose**: Remove noise fast, discover structure
- **Pruning**: Aggressive, every step
- **Threshold**: 0.001 (2-5x initial amplitude range)
- **Frequency addition**: ALLOWED when gradient is strong (>0.1)
- **Behavior**: Most random frequencies get pruned quickly

**Phase 2 (20-100% of training): REFINEMENT**
- **Purpose**: Stabilize spectra, converge
- **Pruning**: Gentle, every 1000 steps
- **Threshold**: 0.0001 (10% of early threshold)
- **Frequency addition**: NOT ALLOWED
- **Behavior**: Only adjust amplitudes, maintain sparsity

### 3. Amplitude Normalization

**Why Normalize?**
- Amplitudes grow/shrink unpredictably during training
- Without normalization: runaway amplitudes, unstable pruning thresholds
- Normalization keeps thresholds meaningful

**When**: After every spectrum update

**Method**: L2 (Euclidean) normalization
```
norm = √(Σ amp_i²)
amp_i ← amp_i / norm
```

### 4. Frequency Addition (Early Only)

**Why Allow Adding Frequencies?**
- Random initialization might miss important frequencies
- Model needs ability to discover new semantic factors
- Cluster structures don't appear immediately

**When**: First 20% of training only

**Condition**: Gradient magnitude > 0.1 on unused frequency

**Behavior**:
- Searches for unused frequency with strongest gradient
- Adds with small amplitude (0.001)
- Allows "late emergence" of polysemy

---

## The Danger: Timing is Critical

### ⚠️ Prune Too Aggressively Before Structure Emerges → Model Killed

If you prune at threshold 0.05 when amplitudes are still 0.001-0.003:
- **Result**: Everything gets pruned immediately
- **Model**: Dead, no learning possible

### ⚠️ Prune Too Late → Noise Dominates

If you wait until epoch 50% to start pruning:
- **Result**: Random frequencies get reinforced by noise
- **Model**: Never converges, interpretability ruined

### ✅ Correct Timing: 0-20% / 20-100% Split

**Early (0-20%)**: Aggressive pruning removes noise before it gets reinforced
**Late (20-100%)**: Gentle pruning allows convergence

This mirrors:
- Neural nets: Early exploration, then stabilization
- Biology: Synaptic pruning
- Compressive sensing: Hard thresholding → refinement

---

## Implementation Details

### Dense Initialization (SpectralWord.js:49-71)

```javascript
initializeWord(wordId) {
  const frequencies = [];
  const amplitudes = [];

  // Start DENSE (all words get maxFrequencies)
  for (let i = 0; i < this.maxFrequencies; i++) {
    let freq = randomUniqueFrequency();
    frequencies.push(freq);

    // CRITICAL: Tiny initial amplitudes [0.0005, 0.003]
    amplitudes.push(Math.random() * 0.0025 + 0.0005);
  }

  this.spectra.set(wordId, { frequencies, amplitudes });
}
```

**Key Parameters:**
- `maxFrequencies`: Usually 5-8
- Initial amplitude range: [0.0005, 0.003]
- **Ratio**: pruning threshold / initial amplitude = 0.001 / 0.002 = ~0.5x

### Two-Phase Pruning (SpectralWord.js:125-210)

```javascript
updateSpectrum(wordId, gradient, learningRate, sparsityPenalty, pruningConfig) {
  const progress = epoch / maxEpochs; // 0 to 1

  // Update amplitudes (gradient descent + sparsity penalty)
  for (let i = 0; i < spectrum.amplitudes.length; i++) {
    const newAmp = amp - learningRate * grad[freq];
    spectrum.amplitudes[i] = Math.max(0, newAmp - sparsityPenalty);
  }

  // PHASE 1 (0-20%): Add frequencies if gradient is strong
  if (progress < 0.2 && spectrum.frequencies.length < maxFrequencies) {
    const bestFreq = findStrongestGradient(gradient, unusedFrequencies);
    if (gradientMag > 0.1) {
      spectrum.frequencies.push(bestFreq);
      spectrum.amplitudes.push(0.001); // Small initial amplitude
    }
  }

  // TWO-PHASE PRUNING
  if (progress < 0.2) {
    // PHASE 1: Aggressive, every step
    pruneThreshold = 0.001;
    pruneNow();
  } else {
    // PHASE 2: Gentle, every 1000 steps
    if (updateStep % 1000 === 0) {
      pruneThreshold = 0.0001;
      pruneNow();
    }
  }
}
```

### Amplitude Normalization (SpectralWord.js:241-263)

```javascript
normalizeSpectrum(wordId, method = 'euclidean') {
  const spectrum = this.getSpectrum(wordId);

  // L2 normalization
  const norm = Math.sqrt(spectrum.amplitudes.reduce((sum, amp) => sum + amp * amp, 0));

  if (norm > 1e-10) {
    for (let i = 0; i < spectrum.amplitudes.length; i++) {
      spectrum.amplitudes[i] /= norm;
    }
  }
}
```

**When Called**: After every `updateSpectrum()` call

---

## Training Phases

### Epoch 0-2 (0-20%): EXPLORATION Phase

**What Happens:**
- Words start with 5-8 random frequencies, tiny amplitudes
- Gradient signals emerge from context measurements
- Meaningful frequencies get reinforced (amplitudes grow)
- Noise frequencies stay weak (amplitudes stay tiny)
- Aggressive pruning removes weak frequencies every step
- Strong gradients can add new frequencies

**Expected Output:**
```
Epoch 1/10 (EXPLORATION phase, 10% complete)
  Avg Loss: 0.456789
  Avg Sparsity: 4.23 active frequencies
  Frequencies added: 234
  Frequencies pruned: 1456
  Negative updates: 3421
```

**Observations:**
- More pruning than adding (1456 vs 234)
- Sparsity dropping from initial 5-8 to 4.23
- Model discovering structure

### Epoch 3-10 (20-100%): REFINEMENT Phase

**What Happens:**
- No new frequencies added
- Gentle pruning every 1000 steps
- Focus on adjusting amplitudes
- Spectra stabilize and converge
- Polysemy emerges (if data supports it)

**Expected Output:**
```
Epoch 5/10 (REFINEMENT phase, 50% complete)
  Avg Loss: 0.234567
  Avg Sparsity: 3.89 active frequencies
  Negative updates: 3421
```

**Observations:**
- No frequency addition/pruning stats (only printed in exploration)
- Sparsity stabilizing around 3-4 frequencies
- Loss decreasing

---

## Parameter Guidelines

### Initial Amplitude Range

**Recommended**: [0.0005, 0.003]

**Too small** (<0.0001):
- Gradients too weak
- Training very slow
- May underflow to zero

**Too large** (>0.01):
- Random signals dominate early
- Model learns noise
- Poor interpretability

**Rule of thumb**:
```
initial_amp_max ≈ 2-5 × early_prune_threshold
```

### Pruning Thresholds

**Early threshold**: 0.001
- Should be 2-5x the initial amplitude range
- Removes noise without killing meaningful signals

**Late threshold**: 0.0001
- Should be ~10% of early threshold
- Only removes true noise, preserves weak but meaningful frequencies

**Adjustment**:
- If too much gets pruned early → lower early threshold
- If noise persists late → raise late threshold

### Phase Transition Point

**Recommended**: 20% of training

**Too early** (<10%):
- Structure hasn't emerged yet
- Model doesn't know what's meaningful
- Risk of premature commitment

**Too late** (>40%):
- Noise gets reinforced
- Hard to remove later
- Interpretability damaged

**Safe range**: 15-25% of training

### Frequency Addition Threshold

**Recommended**: 0.1

**Too low** (<0.05):
- Adds too many frequencies
- Model becomes dense again
- Defeats purpose of sparsity

**Too high** (>0.2):
- Never adds frequencies
- Misses important semantic factors
- Polysemy can't emerge if missed in init

---

## Expected Behavior

### Simple Monosemous Words (e.g., "aardvark", "zeitgeist")

**Initialization**: 5-8 random frequencies, tiny amplitudes
**Exploration**: Most frequencies pruned (1-2 remain)
**Refinement**: Amplitudes stabilize, single dominant frequency
**Final**: 1-2 active frequencies

### Normal Words (e.g., "book", "run", "happy")

**Initialization**: 5-8 random frequencies
**Exploration**: Some pruning, some reinforcement
**Refinement**: 3-5 frequencies stabilize
**Final**: 3-5 active frequencies (moderate polysemy)

### Polysemous Words (e.g., "bank", "set", "play")

**Initialization**: 5-8 random frequencies
**Exploration**: Multiple frequencies get reinforced by different contexts
**Refinement**: Different senses emerge as distinct frequency peaks
**Final**: 5-8 active frequencies (high polysemy)

### Frequency Clusters

**Good interpretability** (after training):
```
Frequency 37:
  bank (0.81)     ← Financial sense
  money (0.78)
  loan (0.71)
  account (0.65)

Frequency 412:
  bank (0.74)     ← River sense
  river (0.69)
  stream (0.63)
  shore (0.58)
```

**Poor interpretability** (training failed):
```
Frequency 89:
  the (0.34)      ← Random unrelated words
  purple (0.22)
  running (0.15)
  bank (0.12)
```

---

## Debugging Issues

### Issue: All words have 1 frequency

**Cause**: Pruning too aggressive or sparsity penalty too high

**Fix**:
- Lower early pruning threshold (0.001 → 0.0005)
- Lower sparsity penalty (0.003 → 0.001)
- Check initial amplitudes aren't too small

### Issue: All words stay at maxFrequencies

**Cause**: Pruning not aggressive enough

**Fix**:
- Raise early pruning threshold (0.001 → 0.002)
- Increase sparsity penalty (0.003 → 0.005)
- Check phase transition is happening (verify progress < 0.2 logic)

### Issue: Frequencies not interpretable

**Cause**: Noise survived exploration phase

**Fix**:
- More aggressive early pruning
- Longer exploration phase (20% → 30%)
- Check amplitude normalization is working

### Issue: Polysemy not emerging

**Cause**:
1. Frequencies pruned before polysemy discovered
2. Not enough data with different senses
3. Frequency addition threshold too high

**Fix**:
1. Gentler early pruning
2. Add more diverse training data
3. Lower frequency addition threshold (0.1 → 0.05)
4. Check known polysemous words with debug utils

---

## Summary

The dense initialization + two-phase pruning strategy allows CSS to:

1. ✅ **Explore** many semantic directions initially
2. ✅ **Discover** meaningful structure through reinforcement
3. ✅ **Prune** noise aggressively in exploration phase
4. ✅ **Stabilize** spectra in refinement phase
5. ✅ **Emerge** polysemy naturally from data
6. ✅ **Adapt** sparsity per word (1-8 frequencies)

**Key insight**: Random dense initialization is NOT a problem if:
- Amplitudes are tiny (0.0005-0.003)
- Pruning is aggressive early (threshold 0.001, every step)
- Pruning becomes gentle later (threshold 0.0001, every 1000 steps)
- Amplitudes are normalized to prevent runaway growth

This approach is **principled**, **theoretically motivated**, and **empirically effective** for discovering sparse, interpretable semantic representations.
