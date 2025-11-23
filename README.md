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
│   ├── core/
│   │   ├── SpectralWord.js       # Word spectrum representation
│   │   ├── ContextMeasurement.js # Context as measurement operator
│   │   └── CSSTrainer.js         # Training algorithm
│   ├── preprocessing/
│   │   └── tokenizer.js          # Text tokenization
│   ├── utils/
│   │   ├── matrix.js             # Linear algebra utilities
│   │   └── ModelPersistence.js   # Save/load models
│   ├── analysis/
│   │   ├── PolysemyAnalyzer.js   # Polysemy analysis
│   │   └── StabilityAnalyzer.js  # Training stability analysis
│   ├── download_parquet.js       # Download Parquet files from HuggingFace
│   ├── index.js                  # Main demo
│   ├── test.js                   # Test suite
│   └── train.js                  # Training script
├── docs/
│   ├── ABSTRACT.md               # Research abstract
│   ├── TRAINING.md               # Paradigm explanation
│   └── INSTRUCTIONS.md           # Detailed instructions
├── IMPLEMENTATION_GUIDE.md       # Code implementation reference
├── TRAINING_GUIDE.md             # Training guide
└── package.json
```

## Installation

```bash
npm install
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

### Run Tests:
```bash
npm test
```

### Large-Scale Training:

**Step 1: Download data**
```bash
npm run download
```

Downloads Parquet files directly from HuggingFace (auto-resumes on next run).

**Step 2: Process data** (TODO: create processing script to read Parquet and build corpus cache)

**Step 3: Train model**
```bash
npm run train
```

This will:
1. Load cached corpus from `./checkpoints/corpus_cache.json`
2. Train CSS model on real-world text
3. Save trained model with timestamp to `./models/`
4. Show real-time progress tracking

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed training instructions.

## Example Output

```
Word: "bank" (ID: 42)
  Active Frequencies: 1
  Dominant Modes:
    1. Freq=12, Amp=0.1374
```

The word "bank" can exhibit multiple active frequencies when trained on larger datasets, demonstrating CSS's ability to capture multiple semantic modes (financial bank vs. river bank).

## API

### CSSTrainer

```javascript
import { CSSTrainer } from './core/CSSTrainer.js';

const trainer = new CSSTrainer({
  frequencyDim: 100,       // Total frequency space
  maxFrequencies: 5,       // Max active frequencies per word
  windowSize: 2,           // Context window size
  learningRate: 0.05,      // Learning rate
  sparsityPenalty: 0.002,  // L1 sparsity penalty
  epochs: 15,              // Training epochs
  batchSize: 50,           // Batch size
  negativeCount: 5,        // Number of negative samples per positive
  margin: 0.5,             // Contrastive margin
  updateNegatives: true    // Update negative word spectra
});

trainer.initialize(vocabSize);
trainer.train(corpus);

// Get word spectrum
const spectrum = trainer.getWordSpectrum(wordId);

// Find similar words
const similar = trainer.findSimilar(wordId, topK=5);

// Export model
const model = trainer.exportModel();
```

### SpectralWord

```javascript
import { SpectralWord } from './core/SpectralWord.js';

const spectralWord = new SpectralWord(vocabSize, maxFreqs, freqDim);

// Get spectrum
const spectrum = spectralWord.getSpectrum(wordId);

// Convert to dense vector
const dense = spectralWord.toDenseVector(wordId);

// Get sparsity
const sparsity = spectralWord.getSparsity(wordId);
```

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

## Future Directions

- [ ] Port to Python with NumPy/PyTorch
- [ ] Scale to large corpora (Wikipedia, Common Crawl)
- [ ] Add context-sensitive disambiguation
- [ ] Implement composition operators
- [ ] Benchmark against Word2Vec/GloVe
- [ ] Visualize frequency spectra
- [ ] Add sense clustering algorithms

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
