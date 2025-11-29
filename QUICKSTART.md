# CSS Quick Start Guide

Get up and running with Compressive Semantic Spectra (CSS) in 5 minutes.

## Prerequisites

- Node.js (v16 or later)
- Python 3 (for reading Parquet files)
- At least 8GB RAM for training
- ~5GB disk space for training data

## Installation

```bash
cd embeddings-js
npm install
```

## Complete Training Pipeline

### Step 1: Download Training Data

```bash
npm run download
```

This downloads Parquet files from HuggingFace. Files are saved to `./data/parquet/`.

**Time**: ~10-30 minutes depending on your connection
**Output**: Several `.parquet` files (~2-5GB total)

### Step 2: Prepare Vocabulary

```bash
npm run prepare:frequency-scaled
```

This explores your corpus and builds a complete vocabulary with properly initialized word spectra.

**Why frequency-scaled?** It prevents common words like "the", "and" from dominating early training.

**Time**: ~5-15 minutes
**Output**: `./data/vocabulary.json` (~50-200MB)

**Other options:**
```bash
npm run prepare:gaussian        # Standard Gaussian initialization
npm run prepare:uniform         # Simple uniform initialization
```

### Step 3: Train the Model

```bash
npm run train
```

Training is automatic and resumable. Press Ctrl+C anytime and run the same command to resume.

**Time**: ~1-4 hours for 10,000-50,000 documents
**Interruption**: Press **Ctrl+C** to stop training gracefully (saves progress automatically)
**Output**:
- `./data/vocabulary.json` (updated with learned spectra)
- `./data/training_state.json` (progress tracking)
- `./data/snapshots/snapshot_*.json` (model snapshots every 5000 docs)

**Important**: You can safely interrupt training with Ctrl+C at any time. The script will save your progress and you can resume later by running `npm run train` again.

**What you'll see:**
```
======================================================================
CSS (COMPRESSIVE SEMANTIC SPECTROSCOPY) TRAINING
======================================================================

Configuration:
  Frequency dimension: 256
  Max frequencies/word: 16
  Window size: 2

Processing file 1/3: train-00000-of-00003.parquet
  [EXPLORATION] Docs: 1000, Vocab: 15234, 45.2 docs/sec
  Checkpoint saved (Ctrl+C safe)
  📸 Snapshot 1 saved: snapshot_0001_docs5000.json
```

**If you press Ctrl+C:**
```
⚠️  Training interrupted by SIGINT (Ctrl+C)
Saving current progress...

  Saving vocabulary...
  ✓ Vocabulary saved
  Saving training state...
  ✓ Training state saved

✅ Progress saved successfully!
  Total documents processed: 7234
  Vocabulary size: 18456

You can resume training by running: npm run train
```

### Step 4: Analyze Polysemy

Once you have at least one snapshot, test if the model learned multiple word senses:

```bash
# Analyze a single snapshot
npm run analyze -- ./data/snapshots/snapshot_0001_docs5000.json bank oxygen

# Or analyze evolution across all snapshots
npm run analyze:all -- bank oxygen
```

**Time**: ~1-2 minutes
**Output**: Comprehensive polysemy report with 5 tests

**What you'll see:**
```
======================================================================
TEST 1: MULTIPLE DOMINANT PEAKS
======================================================================

BANK:
  Dominant peaks (amplitude > 0.3):
    1. Freq 12: amplitude 0.4201
    2. Freq 87: amplitude 0.3892
    3. Freq 203: amplitude 0.4156
  Total peaks: 3
  Potentially polysemous: YES ✓

======================================================================
POLYSEMY DETECTION SUMMARY
======================================================================

Polysemy Score: 4/4
Confidence: 100.0%

✓ POLYSEMY DETECTED: Model successfully learned multiple senses
```

## Quick Demo (No Training)

Want to see CSS in action without training? Run the demo:

```bash
npm start
```

This uses a small sample corpus to demonstrate the algorithm.

## Common Polysemous Words to Test

Try these words in your analysis:

**High polysemy:**
- `bank` - financial / river
- `plant` - organism / factory
- `bat` - animal / equipment
- `spring` - season / metal / water
- `rock` - stone / music / motion

**Monosemous (for comparison):**
- `oxygen` - chemical element
- `hydrogen` - chemical element
- `photosynthesis` - biological process

## Workflow Summary

```
┌──────────────┐
│   Download   │  npm run download
│    Data      │
└──────┬───────┘
       │
       v
┌──────────────┐
│   Prepare    │  npm run prepare:frequency-scaled
│  Vocabulary  │
└──────┬───────┘
       │
       v
┌──────────────┐
│    Train     │  npm run train
│    Model     │  (resumable, Ctrl+C safe)
└──────┬───────┘
       │
       v
┌──────────────┐
│   Analyze    │  npm run analyze:all -- bank oxygen
│  Polysemy    │
└──────────────┘
```

## Troubleshooting

### Problem: "No parquet files found"
**Solution**: Run `npm run download` first

### Problem: "Vocabulary is empty"
**Solution**: Run `npm run prepare:frequency-scaled` before training

### Problem: Training is slow
**Solution**: This is normal for JavaScript. Consider reducing corpus size or wait for Python port.

### Problem: "Word not found in vocabulary"
**Solution**: The word didn't appear in your training corpus. Try:
- More training data
- Lower `minFrequency` in `src/prepare_vocabulary.js`

### Problem: All polysemy tests fail
**Solution**: Model needs more training:
- Train on more documents (at least 10,000+)
- Check that common polysemous words appear frequently
- Wait for later snapshots (polysemy emerges gradually)

### Problem: Out of memory
**Solution**: The training script already allocates 12GB. If still failing:
- Close other applications
- Process fewer files at once
- Reduce `frequencyDim` in `src/train.js`

## Next Steps

Once you have a trained model:

1. **Analyze different words**: Try various polysemous words to see how well the model captures different senses

2. **Compare snapshots**: Use `npm run stability` to track how spectra change over training

3. **Adjust hyperparameters**: Edit `src/train.js` CONFIG section to tune the model:
   - Increase `frequencyDim` for more semantic dimensions
   - Increase `maxFrequencies` to allow more peaks per word
   - Adjust `sparsityPenalty` to control sparsity
   - Modify `windowSize` for larger/smaller context windows

4. **Read the documentation**:
   - [TESTS.md](TESTS.md) - Deep dive into polysemy tests
   - [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Detailed training instructions
   - [docs/ABSTRACT.md](docs/ABSTRACT.md) - Theoretical background

## Performance Tips

### Faster Training
- Use SSD for `./data/` directory
- Close resource-heavy applications
- Use `frequency-scaled` initialization (better convergence)

### Better Results
- More training data = better polysemy detection
- Don't interrupt training during exploration phase (first 20%)
- Let training complete at least one snapshot (5000 docs minimum)
- Compare multiple snapshots to see temporal evolution

### Monitoring
- Watch for "Checkpoint saved" messages (every 1000 docs)
- Check vocabulary size growth (should stabilize eventually)
- Monitor docs/sec throughput (aim for 30-50 docs/sec)

## Example Session

Complete workflow from scratch:

```bash
# 1. Download data (one time)
npm run download
# Wait ~15 minutes

# 2. Prepare vocabulary (one time)
npm run prepare:frequency-scaled
# Wait ~10 minutes

# 3. Train model (can resume)
npm run train
# Wait ~2 hours for 20,000 documents
# Press Ctrl+C if needed

# 4. Analyze results
npm run analyze:all -- bank oxygen
npm run analyze:all -- plant stone
npm run analyze:all -- bat rock

# 5. Check stability
npm run stability -- ./data/snapshots/snapshot_0001_docs5000.json ./data/snapshots/snapshot_0002_docs10000.json
```

## Getting Help

- Check [README.md](README.md) for overview
- Read [TESTS.md](TESTS.md) for analysis details
- See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for advanced training
- Open an issue on GitHub for bugs or questions

## Success Criteria

You know your model is working when:

✅ Training completes multiple checkpoints without errors
✅ Vocabulary size stabilizes (stops growing rapidly)
✅ Snapshots show decreasing sparsity over time
✅ Polysemous words score >60% confidence in analysis
✅ Monosemous words show 1 dominant peak
✅ Context clustering shows clear separation (silhouette >0.3)

Happy training! 🎉
