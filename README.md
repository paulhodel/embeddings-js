# Compressive Semantic Spectroscopy (CSS)

A JavaScript implementation of a novel paradigm for learning word meaning as sparse spectra rather than dense vectors.

## Overview

Unlike traditional word embeddings (Word2Vec, GloVe) that represent words as dense vectors, CSS represents each word as a **sparse spectrum** of semantic frequencies:

```
S_w(ω) = Σ A_k * δ(ω - ω_k)
```

Where:
- **ω_k**: Active semantic frequencies (distinct modes/senses)
- **A_k**: Amplitudes (relevance/strength)

**Note**: This simplified version uses real-valued amplitudes only. Phase information has been intentionally dropped for:
- Simpler mathematics and faster computation
- Easier interpretation
- Focus on core CSS benefits: polysemy, sparsity, and context filtering

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
│   ├── data/
│   │   └── HuggingFaceStreamer.js # HuggingFace dataset streaming
│   ├── preprocessing/
│   │   └── tokenizer.js          # Text tokenization
│   ├── utils/
│   │   ├── matrix.js             # Linear algebra utilities
│   │   └── ModelPersistence.js   # Save/load models
│   ├── index.js                  # Main demo
│   ├── test.js                   # Test suite
│   └── train.js                  # Large-scale training script
├── ABSTRACT.md                   # Research abstract
├── TRAINING.md                   # Paradigm explanation
├── TRAINING_GUIDE.md             # Large-scale training guide
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
```bash
npm run train
```

This will:
1. Stream data from HuggingFace's fineweb-edu-100b dataset
2. Build vocabulary from streaming data
3. Train CSS model on real-world text
4. Save trained model to `./models/css_model.json`
5. Show real-time progress tracking

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
  title={Compressive Semantic Spectroscopy: A Sparse Spectral Framework for Learning Word Meaning},
  author={[Author]},
  year={2025},
  url={https://github.com/[username]/embeddings-js}
}
```
