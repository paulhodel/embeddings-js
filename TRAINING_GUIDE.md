# CSS Training Guide

## Large-Scale Training with HuggingFace Dataset Streaming

This guide explains how to train CSS models on large datasets using streaming from HuggingFace.

---

## Quick Start

```bash
# Install dependencies
npm install

# Run training
npm run train
```

---

## Features

### 🌊 **Streaming Data**
- Fetches data directly from HuggingFace datasets API
- No need to download the entire dataset
- Memory-efficient: processes data in batches

### 📊 **Progress Tracking**
- Real-time statistics: documents/second, tokens/second
- Dataset progress percentage
- Elapsed time tracking

### 💾 **Model Persistence**
- Automatic model saving after training
- JSON format for easy inspection
- Automatic backup of existing models
- Training metadata saved alongside model

### ⏸️ **Checkpoint & Resume** (NEW!)
- Automatic checkpointing every N batches
- Resume training if interrupted (Ctrl+C)
- Saves vocabulary, documents, and progress
- No data loss if training stops unexpectedly

### 🎯 **Configurable**
- Easy configuration in `src/train.js`
- Adjust vocabulary size, model parameters, training settings

---

## Configuration

Edit the `CONFIG` object in `src/train.js`:

### Dataset Settings

```javascript
dataset: 'karpathy/fineweb-edu-100b-shuffle',  // HuggingFace dataset
split: 'train',                                 // Dataset split
maxBatches: 100,                                // Limit batches (null = unlimited)
batchSize: 50,                                  // Documents per fetch
maxDocLength: 5000,                             // Max chars per document
```

### Vocabulary Settings

```javascript
minWordFreq: 5,          // Minimum word frequency to include
maxVocabSize: 50000,     // Maximum vocabulary size
```

### Model Settings

```javascript
frequencyDim: 200,       // Total frequency space dimension
maxFrequencies: 8,       // Max active frequencies per word
windowSize: 3,           // Context window size
learningRate: 0.03,      // Learning rate
sparsityPenalty: 0.003,  // L1 sparsity penalty
negativeCount: 5,        // Negative samples per positive
margin: 0.5,             // Contrastive margin
```

### Training Settings

```javascript
epochs: 5,                    // Number of training epochs
trainingBatchSize: 100,       // Batch size for training
```

### Checkpointing

```javascript
saveCheckpointEvery: 20,      // Save checkpoint every 20 batches
resumeFromCheckpoint: true,   // Auto-resume if checkpoint exists
checkpointPath: './checkpoints/training_checkpoint.json',
modelPath: './models/css_model.json',
```

---

## Training Process

### Phase 1: Stream Data & Build Vocabulary
1. Connects to HuggingFace datasets API
2. Streams batches of text documents
3. Builds vocabulary from streamed data
4. Tracks progress (documents, tokens, speed)

### Phase 2: Convert to Corpus
- Converts text documents to word ID sequences
- Filters out empty documents
- Prepares corpus for training

### Phase 3: Train CSS Model
- Initializes CSS trainer with configuration
- Trains using contrastive learning with negative sampling
- Shows epoch-by-epoch progress

### Phase 4: Save Model
- Backs up existing model (if present)
- Saves trained model to disk
- Saves training metadata

---

## Output Files

After training, you'll have:

### `./models/css_model.json`
Complete trained model including:
- Configuration
- Vocabulary mappings
- Word spectra (frequencies, amplitudes, phases)
- Training statistics

### `./models/css_model.metadata.json`
Training metadata:
- Timestamp
- Configuration used
- Training statistics (documents, tokens, time)
- Dataset progress

### `./models/css_model.backup.*.json`
Automatic backups of previous models

---

## Example Output

```
██████████████████████████████████████████████████████████████████████
COMPRESSIVE SEMANTIC SPECTROSCOPY - LARGE SCALE TRAINING
██████████████████████████████████████████████████████████████████████

Configuration:
  Dataset: karpathy/fineweb-edu-100b-shuffle
  Max batches: 100
  Batch size: 50
  Frequency dim: 200
  Max frequencies: 8
  Vocabulary size: up to 50,000

Dataset has 200,000,000 rows in train split

▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
PHASE 1: STREAMING DATA & BUILDING VOCABULARY
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

Streaming data from HuggingFace...

======================================================================
TRAINING PROGRESS
======================================================================
Batches processed: 50
Documents: 2,500
Tokens: 458,923
Elapsed time: 45.3s
Speed: 55.2 docs/s, 10134 tokens/s
Dataset progress: 2,500/200,000,000 (0.00%)
======================================================================

...

▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
PHASE 3: TRAINING CSS MODEL
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

Initialized CSS model:
  Vocabulary size: 45,821
  Frequency dimension: 200
  Max frequencies per word: 8
  Negative samples per positive: 5

=== Phase 2: Sparse Reconstruction (with Negative Sampling) ===
  Epoch 1/5
    Avg Loss: 0.482351
    Avg Sparsity: 5.23 active frequencies
    Negative updates: 45821
  Epoch 2/5
    Avg Loss: 0.465210
    Avg Sparsity: 5.01 active frequencies
    Negative updates: 45821
...

▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
PHASE 4: SAVING MODEL
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

✓ Backup created: ./models/css_model.backup.2025-01-23T01-15-30.json

Saving model to ./models/css_model.json...
✓ Model saved (12.45 MB)
  Words: 45821
  Avg sparsity: 5.01 frequencies/word

✓ Metadata saved: ./models/css_model.metadata.json

██████████████████████████████████████████████████████████████████████
TRAINING COMPLETE!
██████████████████████████████████████████████████████████████████████

Total documents: 5,000
Total tokens: 892,345
Vocabulary size: 45,821
Training time: 234.5s

Model saved to: ./models/css_model.json

██████████████████████████████████████████████████████████████████████
```

---

## Loading a Trained Model

```javascript
import { ModelPersistence } from './utils/ModelPersistence.js';
import { CSSTrainer } from './core/CSSTrainer.js';
import { Tokenizer } from './preprocessing/tokenizer.js';

// Load model
const loaded = ModelPersistence.loadModel('./models/css_model.json');

// Reconstruct trainer
const trainer = new CSSTrainer(loaded.config);
trainer.initialize(loaded.modelData.vocabSize);
trainer.importModel(loaded.modelData);

// Reconstruct tokenizer
const tokenizer = new Tokenizer();
tokenizer.vocab = loaded.vocab;
tokenizer.wordFreq = loaded.wordFreq;
// ... rebuild idToWord map

// Now you can use the model
const wordId = tokenizer.wordToId('example');
const spectrum = trainer.getWordSpectrum(wordId);
const similar = trainer.findSimilar(wordId, 10);
```

---

## Checkpointing & Resume

### How It Works

The training script **automatically saves checkpoints** during Phase 1 (data streaming):

1. **Automatic Saving**: Every N batches (default: 20), saves:
   - All downloaded documents
   - Built vocabulary
   - Training progress (batches, documents, tokens)
   - Configuration

2. **Automatic Resume**: If training is interrupted:
   - Next run detects the checkpoint
   - Restores all data and progress
   - Continues from where it left off

3. **Checkpoint Location**: `./checkpoints/training_checkpoint.json`

### Usage

**To pause training:**
- Press `Ctrl+C` to stop
- Checkpoint is already saved automatically

**To resume training:**
- Just run `npm run train` again
- Script will detect checkpoint and resume

**To start fresh:**
- Delete `./checkpoints/training_checkpoint.json`
- Or set `resumeFromCheckpoint: false` in config

### Example

```bash
# Start training
npm run train

# After 20 batches, you'll see:
💾 Saving checkpoint...
✓ Checkpoint saved: ./checkpoints/training_checkpoint.json

# Press Ctrl+C to stop

# Resume later
npm run train

# You'll see:
▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶
RESUMING FROM CHECKPOINT
▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶
Checkpoint from: 2025-01-23T15:30:45.123Z
Phase: streaming
Batches processed: 20
Documents: 1,000
Tokens: 250,000
▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶

✓ Restored 1,000 documents from checkpoint
✓ Restored vocabulary: 5,234 words

# Training continues from batch 21...
```

### What Gets Saved

| Data | Saved | Purpose |
|------|-------|---------|
| Downloaded documents | ✅ | Don't re-download |
| Vocabulary | ✅ | Resume with same vocab |
| Word frequencies | ✅ | For vocab filtering |
| Progress counters | ✅ | Resume from correct position |
| Configuration | ✅ | Ensure consistency |
| Trained model | ❌ | Only saved at end |

**Note**: Checkpoints are saved during **Phase 1 (streaming)** only. Once Phase 3 (training) starts, the model trains to completion.

### Benefits

- ✅ **No data loss** if training is interrupted
- ✅ **Resume anytime** without re-downloading
- ✅ **Save bandwidth** on HuggingFace API
- ✅ **Experiment safely** - pause and adjust config

---

## Tips for Large-Scale Training

### Memory Management
- If running out of memory, reduce `maxBatches` or `batchSize`
- Consider processing data in multiple passes
- Monitor memory usage during training

### Vocabulary Size
- Start with smaller `maxVocabSize` (10k-20k) for faster training
- Increase `minWordFreq` to filter rare words
- Balance vocabulary size vs. coverage

### Training Speed
- Reduce `frequencyDim` for faster training (100-200)
- Reduce `maxFrequencies` for sparser spectra (5-8)
- Use fewer `epochs` (3-5) for initial experiments

### Dataset Streaming
- Set `maxBatches` to a small number (10-50) for quick tests
- Set to `null` for full dataset training
- Monitor the HuggingFace API rate limits

---

## Troubleshooting

### "No matching version found for @huggingface/hub"
- This is expected - we use the HuggingFace datasets API directly via HTTP
- No additional packages needed

### "Error fetching batch: HTTP 404"
- Dataset may not be available or reached end
- Check dataset name and split are correct

### "Out of memory"
- Reduce `maxBatches` or `batchSize`
- Reduce `maxVocabSize`
- Process in smaller chunks

### Slow training
- Check network connection (streaming speed)
- Reduce `frequencyDim` or vocabulary size
- Use smaller dataset for testing

---

## Advanced: Custom Datasets

To use a different HuggingFace dataset:

```javascript
// In src/train.js CONFIG
dataset: 'your-dataset/name',
split: 'train',

// Make sure the dataset has a 'text' field
// Or modify HuggingFaceStreamer.js to extract the correct field
```

---

## Next Steps

After training:
1. Analyze word spectra with `trainer.getWordSpectrum(wordId)`
2. Find similar words with `trainer.findSimilar(wordId, topK)`
3. Explore polysemy by examining multi-peak spectra
4. Export embeddings for downstream tasks
5. Visualize frequency distributions
