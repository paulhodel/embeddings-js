# Compressive Semantic Spectra (CSS)

A Sparse Spectral Framework for Learning Word Meaning via Compressive Reconstruction

## Overview

CSS reframes distributional semantics as a **sparse spectral reconstruction problem**. Unlike traditional dense embeddings (Word2Vec, GloVe), CSS represents each word as a **sparse spectrum** over a fixed frequency basis:

```
S_w(ω) = Σ_{k=1}^K A_k * e^{iφ_k} * δ(ω - ω_k)
```

Where:
- **ω_k ∈ [1, D]**: Active semantic frequency indices (K << D)
- **A_k ∈ ℝ⁺**: Amplitudes (semantic relevance/strength)
- **φ_k ∈ [0, 2π]**: Phases (relational orientation, analogy structure)
- **K**: Sparsity level (typically 2-8 active frequencies per word)
- **D**: Frequency space dimension (typically 100-200)

**Training as inverse problem**: Contexts provide noisy measurements of word spectra. Learning recovers the sparsest spectral signature that jointly explains all contextual observations—analogous to compressive sensing or tomographic reconstruction.

**Two-layer semantic representation**:
1. **Frequency/Amplitude layer**: Core semantic content—which concepts and how strongly (e.g., "bank" activates finance + river frequencies)
2. **Phase layer**: Semantic glue—relational structure, analogies, contextual orientation (e.g., bank:money :: dam:water encoded via phase alignment)

**Note**: This implementation focuses on the frequency/amplitude layer, establishing the core sparse spectral framework. Phase dynamics represent a natural extension for capturing finer relational semantics.

## Key Concepts

### 1. Words as Sparse Spectra
Each word has a few active frequencies corresponding to distinct semantic modes. This naturally handles:
- **Polysemy**: Multiple peaks for different senses
- **Interpretability**: Each frequency is a transparent semantic factor
- **Efficiency**: Sparse representation is compact

### 2. Context as Measurement
Instead of "predicting words from context," each context acts as a **measurement operator** that probes specific frequencies of the word's hidden spectrum.

### 3. Training as Inverse Problem
Learning becomes a global sparse reconstruction problem:
> Find the sparsest spectrum S_w that jointly explains all contextual measurements

This is analogous to **compressive sensing** or **tomographic reconstruction**.

## Project Structure

```
embeddings-js/
├── src/
│   ├── train.js                  # Main training script
│   ├── prepare_vocabulary.js     # Vocabulary preparation (pre-training)
│   ├── vocabulary.js             # Vocabulary management module
│   ├── analyze_polysemy.js       # Comprehensive polysemy analysis
│   ├── analyze_stability.js      # Training stability analysis
│   ├── download_parquet.js       # Download Parquet files from HuggingFace
│   └── index.js                  # Quick demo
├── scripts/
│   └── read_parquet.py           # Python helper to read Parquet files
├── data/
│   ├── parquet/                  # Training data (Parquet files)
│   ├── snapshots/                # Model snapshots during training
│   ├── vocabulary.json           # Pre-built vocabulary with spectra
│   └── training_state.json       # Training progress state
├── docs/
│   ├── ABSTRACT.md               # Research abstract
│   ├── TRAINING.md               # Paradigm explanation
│   ├── MATH.md                   # Mathematical foundations
│   ├── RESEARCH.md               # Research notes
│   └── INSTRUCTIONS.md           # Detailed instructions
├── TESTS.md                      # Polysemy detection tests
├── TRAINING_GUIDE.md             # Training guide
└── package.json
```

## Installation

```bash
npm install
```

## Quick Start

```bash
# 1. Download training data
npm run download

# 2. Prepare vocabulary (recommended: frequency-scaled)
npm run prepare:frequency-scaled

# 3. Train the model
npm run train

# 4. Analyze polysemy
npm run analyze -- ./data/snapshots/snapshot_0001_docs5000.json bank oxygen

# Or analyze evolution across all snapshots
npm run analyze:all -- bank oxygen
```

## Usage

### Quick Demo:
```bash
npm start
```

This will:
1. Build vocabulary from sample corpus
2. Train CSS model using sparse reconstruction
3. Analyze word spectra and semantic similarities
4. Demonstrate polysemy handling

### Large-Scale Training (Recommended Workflow):

**Step 1: Download training data**
```bash
npm run download
```

Downloads Parquet files directly from HuggingFace (auto-resumes on next run). Files are saved to `./data/parquet/`.

**Step 2: Prepare vocabulary**
```bash
node src/prepare_vocabulary.js
```

This explores the entire corpus and pre-builds the vocabulary with properly initialized spectra:
- Scans all Parquet files to count word frequencies
- Filters rare words (configurable minimum frequency)
- Initializes all words with random sparse spectra
- Supports three initialization strategies:
  - `uniform`: Simple uniform distribution [0.0005, 0.003]
  - `gaussian`: Gaussian distribution (mean=0.001, stddev=0.0005)
  - `frequency-scaled`: Gaussian scaled by word frequency (recommended)

Choose initialization strategy:
```bash
# Default (Gaussian)
node src/prepare_vocabulary.js

# Explicit strategy
node src/prepare_vocabulary.js --strategy=frequency-scaled
node src/prepare_vocabulary.js --strategy=uniform
node src/prepare_vocabulary.js --strategy=gaussian
```

**Recommended: frequency-scaled** - This prevents high-frequency words like "the", "and" from dominating early training.

**Step 3: Train model**
```bash
node src/train.js
```

Training features:
- Uses pre-built vocabulary (no lazy initialization)
- Contrastive learning with negative sampling
- Two-phase pruning (exploration → refinement)
- Automatic checkpointing every 1000 documents
- Snapshots saved every 5000 documents for analysis
- Resume support (automatically continues from last checkpoint)

Training progress is saved to:
- `./data/vocabulary.json` - Updated vocabulary with learned spectra
- `./data/training_state.json` - Training progress (resumable)
- `./data/snapshots/` - Model snapshots for temporal analysis

**Step 4: Analyze polysemy**
```bash
# Analyze a single snapshot
node src/analyze_polysemy.js ./data/snapshots/snapshot_0001_docs5000.json bank oxygen

# Analyze evolution across all snapshots
node src/analyze_polysemy.js --all bank oxygen
```

This runs comprehensive polysemy detection tests (see [TESTS.md](TESTS.md) for details):
- Test 1: Multiple dominant peaks
- Test 2: Divergence over training snapshots
- Test 3: Context clustering
- Test 4: Mutual reinforcement
- Test 5: Spectrum stability under substitution

**Step 5: Analyze training stability** (optional)
```bash
# Compare two snapshots
node src/analyze_stability.js ./data/snapshots/snapshot_0001_docs5000.json ./data/snapshots/snapshot_0002_docs10000.json

# Time-series analysis (all snapshots)
node src/analyze_stability.js ./data/snapshots/snapshot_*.json
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed training instructions and [TESTS.md](TESTS.md) for polysemy analysis documentation.

## Available NPM Scripts

Quick reference for all available commands:

### Data Preparation
```bash
npm run download                    # Download Parquet files from HuggingFace
npm run prepare                     # Prepare vocabulary (default: Gaussian)
npm run prepare:frequency-scaled    # Prepare with frequency-scaled strategy (recommended)
npm run prepare:gaussian            # Prepare with Gaussian strategy
npm run prepare:uniform             # Prepare with uniform strategy
```

### Training
```bash
npm run train                       # Start or resume training
npm start                           # Quick demo with sample corpus
```

### Analysis
```bash
npm run analyze -- <snapshot> <word> <comparison>    # Analyze single snapshot
npm run analyze:all -- <word> <comparison>           # Analyze all snapshots
npm run stability -- <snapshot1> <snapshot2>         # Compare stability
```

### Testing
```bash
npm test                            # Run test suite
```

## Example Output

### Training Progress:
```
======================================================================
CSS (COMPRESSIVE SEMANTIC SPECTROSCOPY) TRAINING
======================================================================

Configuration:
  Frequency dimension: 256
  Max frequencies/word: 16
  Window size: 2
  Negative samples: 5
  Initial amplitudes: [0.0005, 0.003]
  Two-phase pruning: 20% exploration

Found 3 parquet file(s)

Processing file 1/3: train-00000-of-00003.parquet
Loading parquet: ./data/parquet/train-00000-of-00003.parquet
  Loaded 1000 documents
  [EXPLORATION] Docs: 1000, Vocab: 15234, 45.2 docs/sec
  Checkpoint saved
  📸 Snapshot 1 saved: snapshot_0001_docs5000.json
     Vocab: 15234, Avg sparsity: 8.34
```

### Polysemy Analysis Output:
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

OXYGEN (comparison):
  Dominant peaks (amplitude > 0.3):
    1. Freq 55: amplitude 0.9342
  Total peaks: 1
  Potentially polysemous: NO ✓

======================================================================
POLYSEMY DETECTION SUMMARY
======================================================================

Word: "bank"

Test Results:
  ✓ Test 1 (Multiple Peaks): PASS - 3 dominant peaks
  ✓ Test 3 (Clustering): PASS - Silhouette 0.4521
  ✓ Test 4 (Mutual Reinforcement): PASS - 18.2% overlap
  ✓ Test 5 (Context Substitution): PASS - Contexts differentiated

Polysemy Score: 4/4
Confidence: 100.0%

✓ POLYSEMY DETECTED: Model successfully learned multiple senses
```

The word "bank" exhibits multiple active frequencies when trained on larger datasets, demonstrating CSS's ability to capture multiple semantic modes (financial bank vs. river bank). See [TESTS.md](TESTS.md) for comprehensive polysemy verification tests.

## Configuration

The training script (`src/train.js`) uses the following configuration:

```javascript
const CONFIG = {
    // Data
    parquetDir: './data/parquet',
    stateFile: './data/training_state.json',
    vocabularyFile: './data/vocabulary.json',

    // CSS Model Architecture
    frequencyDim: 256,              // Total semantic frequency space (N)
    maxFrequencies: 16,             // Max active frequencies per word (K)
    windowSize: 2,                  // Context window (2 words before + 2 after)

    // Training Hyperparameters
    learningRate: 0.03,
    sparsityPenalty: 0.003,         // L1 penalty for sparsity
    negativeCount: 5,               // Negative samples per positive
    margin: 0.5,                    // Contrastive margin
    epochs: 10,

    // Initialization
    initAmpMin: 0.0005,             // Tiny initial amplitudes
    initAmpMax: 0.003,

    // Two-Phase Pruning
    explorationPhase: 0.2,          // First 20% of training
    earlyPruneThreshold: 0.001,     // Aggressive early pruning
    latePruneThreshold: 0.0001,     // Gentle late pruning
    latePruneInterval: 1000,        // Prune every N steps in refinement

    // Progress & Snapshots
    saveEvery: 1000,                // Save checkpoint every N documents
    logEvery: 100,                  // Log progress every N documents
    snapshotEvery: 5000,            // Save snapshot for analysis every N documents
};
```

Key hyperparameters:

- `frequencyDim`: Total frequency space dimension (100-512)
- `maxFrequencies`: Maximum active frequencies per word (8-32)
- `sparsityPenalty`: L1 penalty strength (0.001-0.01)
- `learningRate`: Step size for gradient descent (0.01-0.1)
- `windowSize`: Context window size (1-5)
- `negativeCount`: Number of negative samples per positive example (3-10)
- `margin`: Contrastive margin for triplet loss (0.3-1.0)
- `explorationPhase`: Fraction of training for aggressive pruning (0.1-0.3)

## Advantages over Dense Embeddings

1. **Explicit Polysemy**: Multiple frequency peaks naturally represent different word senses
2. **Interpretability**: Each frequency corresponds to a semantic factor
3. **Sparsity**: Compact representation (3-5 active frequencies vs. 100-300 dimensional vectors)
4. **Compositionality**: Spectral interference allows linear, physically meaningful composition
5. **Principled Framework**: Grounded in signal processing and compressive sensing theory

## Training Process

### Phase 1: Collect Measurements
- Extract context windows from corpus
- Create measurement patterns from context words
- Record all (word, context) observations

### Phase 2: Sparse Reconstruction with Contrastive Learning
- **Positive updates**: Push target word spectrum toward its actual contexts
- **Negative sampling**: Sample random words that don't belong in this context
- **Negative updates**: Push negative word spectra away from unrelated contexts
- Apply gradient descent with L1 sparsity penalty
- Prune near-zero frequencies

This contrastive approach ensures:
- Words align with their true semantic contexts (positive signal)
- Words separate from unrelated contexts (negative signal)
- Better discrimination between semantically similar vs. dissimilar words

## Configuration

Key hyperparameters:

- `frequencyDim`: Total frequency space dimension (50-200)
- `maxFrequencies`: Maximum active frequencies per word (3-10)
- `sparsityPenalty`: L1 penalty strength (0.001-0.01)
- `learningRate`: Step size for gradient descent (0.01-0.1)
- `windowSize`: Context window size (1-5)
- `negativeCount`: Number of negative samples per positive example (3-10)
- `margin`: Contrastive margin for triplet loss (0.3-1.0)
- `updateNegatives`: Whether to update negative word spectra (true/false)

## Features

### Implemented ✓
- [x] Sparse spectral word representations
- [x] Contrastive learning with negative sampling
- [x] Two-phase pruning (exploration → refinement)
- [x] Pre-corpus vocabulary preparation with multiple initialization strategies
- [x] Frequency-scaled initialization to prevent high-frequency word dominance
- [x] Automatic checkpointing and resume support
- [x] Training snapshots for temporal analysis
- [x] Comprehensive polysemy detection (5 independent tests)
- [x] Training stability analysis across snapshots
- [x] Context clustering for sense separation
- [x] Temporal evolution tracking (divergence analysis)
- [x] Real-time training progress monitoring

### Future Directions
- [ ] Port to Python with NumPy/PyTorch for faster training
- [ ] Scale to larger corpora (full Wikipedia, Common Crawl)
- [ ] Context-sensitive disambiguation (dynamic sense selection)
- [ ] Spectral composition operators (word combinations)
- [ ] Benchmark against Word2Vec/GloVe/BERT
- [ ] Interactive visualization of frequency spectra
- [ ] Automatic sense discovery (variable number of senses)
- [ ] Multi-language support
- [ ] Phase dynamics implementation (relational semantics)

## Research Background

This implementation is based on the theoretical framework described in:
- **ABSTRACT.md**: Research paper abstract
- **TRAINING.md**: Detailed paradigm explanation

CSS reframes distributional semantics as a **signal reconstruction problem**, opening new research directions for lightweight semantic models and explicit polysemy handling.

## License

MIT

## Citation

If you use this code in your research, please cite:

```
@software{css2025,
  title={Compressive Semantic Spectra: A Sparse Spectral Framework for Learning Word Meaning},
  author={Paul Hodel},
  year={2025},
  url={https://github.com/paulhodel/embeddings-js}
}
```
