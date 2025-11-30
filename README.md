# Word Embeddings - Comparative Study

An academic project implementing and benchmarking two word embedding models for semantic relationship learning.

## Overview

This project implements two different word embedding approaches to learn vector representations of words. The models are evaluated on their ability to capture semantic relationships through vector arithmetic, such as the classic analogy: `king - man + woman = queen`.

## Project Structure

```
src/
├── reference/          # Reference implementation (MNIST neural network)
│   ├── index.js       # Training script with forward/backward pass
│   ├── utils.js       # Neural network utilities (ReLU, softmax, etc.)
│   └── images.js      # MNIST data loading
│
├── models/            # Word embedding model implementations
│   ├── model1/        # [TO BE DEFINED]
│   └── model2/        # [TO BE DEFINED]
│
├── benchmark/         # Evaluation and comparison tools
│   └── [TO BE IMPLEMENTED]
│
└── index.js          # Main entry point
```

## Models

### Model 1: [TO BE DEFINED]
- **Algorithm**: [TO BE SPECIFIED]
- **Architecture**: [TO BE SPECIFIED]
- **Hyperparameters**: [TO BE SPECIFIED]

### Model 2: [TO BE DEFINED]
- **Algorithm**: [TO BE SPECIFIED]
- **Architecture**: [TO BE SPECIFIED]
- **Hyperparameters**: [TO BE SPECIFIED]

## Code Style Guidelines

Based on the reference implementation (`src/reference/`), this project follows these patterns:

### Performance
- Use manual `for` loops for matrix operations instead of array methods
- Pre-allocate arrays with `new Array().fill()` for zero matrices
- Direct array access over abstractions
- Explicit variable names for clarity

### Structure
- JSDoc comments for all functions with parameter descriptions
- Clear separation of concerns (separate files for utils, data loading, training)
- Mathematical operations explicitly written out with inline comments
- Academic-style documentation explaining the mathematics

### Matrix Representation
- Weight matrices stored as 2D arrays: `[numRows][numCols]`
- Batch processing with gradient accumulation where applicable

## Benchmarking

The models will be evaluated on:

1. **Analogy Tasks**: Semantic relationships like `king - man + woman ≈ queen`
2. **[TO BE DEFINED]**: Additional evaluation metrics
3. **Performance Comparison**: Speed, memory usage, accuracy

## Installation

```bash
npm install
```

## Usage

### Training Model 1
```bash
# [TO BE IMPLEMENTED]
```

### Training Model 2
```bash
# [TO BE IMPLEMENTED]
```

### Running Benchmarks
```bash
# [TO BE IMPLEMENTED]
```

## Dataset

- **Source**: [TO BE SPECIFIED]
- **Size**: [TO BE SPECIFIED]
- **Preprocessing**: [TO BE SPECIFIED]

## Results

[TO BE FILLED AFTER EXPERIMENTS]

## Current Status

### ✅ Completed
- [x] Define Model 1 architecture (see ALGORITHM.md)
- [x] Create comprehensive algorithm documentation
- [x] Implement prepare.js (vocabulary extraction and tokenization)
- [x] Download 5 parquet shards for proof-of-concept

### 🚧 In Progress
- [ ] Fix dictionary.json saving (file too large for JSON.stringify, need streaming approach)
- [ ] Implement train.js with full training loop
- [ ] Implement utils.js (matrix operations, activations)

### 📋 TODO
- [ ] Complete Model 1 implementation
- [ ] Define Model 2 architecture
- [ ] Implement Model 2
- [ ] Create benchmark suite (analogy tests, similarity tests)
- [ ] Train Model 1 on full dataset (197 shards)
- [ ] Train Model 2 on full dataset
- [ ] Evaluate and compare results
- [ ] Write research findings

## References

[TO BE ADDED]

---

**Note**: This is an academic project focused on understanding and comparing word embedding approaches.
