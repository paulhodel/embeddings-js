/**
 * Training Script - Train CSS model on cached corpus
 *
 * Usage:
 *   npm run train
 *
 * Prerequisites:
 *   Run `npm run prepare-data` first to download and cache corpus
 *
 * This script:
 * 1. Loads cached corpus (no HuggingFace download)
 * 2. Trains CSS model
 * 3. Saves model with timestamp
 * 4. Saves periodic snapshots
 *
 * Can be run MULTIPLE TIMES with different hyperparameters!
 */

import { Tokenizer } from './preprocessing/tokenizer.js';
import { CSSTrainer } from './core/CSSTrainer.js';
import { ModelPersistence } from './utils/ModelPersistence.js';
import fs from 'fs';

// ============================================
// CONFIGURATION
// ============================================

const CONFIG = {
  // Input
  corpusCachePath: './checkpoints/corpus_cache.json',

  // Model Architecture
  frequencyDim: 200,            // Total frequency space
  maxFrequencies: 8,            // Max active frequencies per word
  windowSize: 3,                // Context window size

  // Training Hyperparameters
  learningRate: 0.03,           // Learning rate
  sparsityPenalty: 0.003,       // L1 sparsity penalty
  epochs: 10,                   // Number of training epochs
  batchSize: 100,               // Batch size for training

  // Contrastive Learning
  negativeCount: 5,             // Negative samples per positive
  margin: 0.5,                  // Contrastive margin
  updateNegatives: true,        // Update negative words

  // Output
  modelDir: './models',
  useTimestampedModels: true,   // Add timestamp to model filenames

  // Snapshots (for stability analysis)
  saveSnapshotEvery: 1,         // Save snapshot every N epochs (0=disable)
  snapshotDir: './models/snapshots'
};

// ============================================
// LOAD CORPUS
// ============================================

function loadCorpusCache(config) {
  if (!fs.existsSync(config.corpusCachePath)) {
    console.error(`\n❌ Corpus cache not found: ${config.corpusCachePath}`);
    console.error('\nPlease run: npm run prepare-data\n');
    process.exit(1);
  }

  try {
    console.log('Loading corpus cache...');
    const cache = ModelPersistence.loadCheckpoint(config.corpusCachePath);

    if (!cache || !cache.corpus || !cache.vocab) {
      throw new Error('Invalid corpus cache format');
    }

    console.log(`✓ Corpus loaded from: ${config.corpusCachePath}`);
    console.log(`  Cached: ${cache.timestamp}`);
    console.log(`  Documents: ${cache.corpusSize.toLocaleString()}`);
    console.log(`  Vocabulary: ${cache.vocabSize.toLocaleString()} words\n`);

    return cache;
  } catch (error) {
    console.error(`\n❌ Error loading corpus cache: ${error.message}`);
    console.error('\nPlease run: npm run prepare-data\n');
    process.exit(1);
  }
}

// ============================================
// MAIN TRAINING FUNCTION
// ============================================

async function train() {
  console.log('\n' + '█'.repeat(70));
  console.log('CSS MODEL TRAINING');
  console.log('█'.repeat(70) + '\n');

  // Load corpus cache
  const corpusCache = loadCorpusCache(CONFIG);

  // Reconstruct tokenizer
  const tokenizer = new Tokenizer();
  tokenizer.vocab = new Map(Object.entries(corpusCache.vocab));
  tokenizer.wordFreq = new Map(Object.entries(corpusCache.wordFreq));
  tokenizer.nextId = tokenizer.vocab.size;

  for (const [word, id] of tokenizer.vocab.entries()) {
    tokenizer.idToWord.set(id, word);
  }

  const corpus = corpusCache.corpus;

  // Display configuration
  console.log('Training Configuration:');
  console.log(`  Frequency dimension: ${CONFIG.frequencyDim}`);
  console.log(`  Max frequencies per word: ${CONFIG.maxFrequencies}`);
  console.log(`  Window size: ${CONFIG.windowSize}`);
  console.log(`  Learning rate: ${CONFIG.learningRate}`);
  console.log(`  Sparsity penalty: ${CONFIG.sparsityPenalty}`);
  console.log(`  Epochs: ${CONFIG.epochs}`);
  console.log(`  Batch size: ${CONFIG.batchSize}`);
  console.log(`  Negative samples: ${CONFIG.negativeCount}`);
  console.log('');

  // ============================================
  // INITIALIZE MODEL
  // ============================================

  console.log('Initializing CSS model...\n');

  const trainer = new CSSTrainer({
    frequencyDim: CONFIG.frequencyDim,
    maxFrequencies: CONFIG.maxFrequencies,
    windowSize: CONFIG.windowSize,
    learningRate: CONFIG.learningRate,
    sparsityPenalty: CONFIG.sparsityPenalty,
    epochs: CONFIG.epochs,
    batchSize: CONFIG.batchSize,
    negativeCount: CONFIG.negativeCount,
    margin: CONFIG.margin,
    updateNegatives: CONFIG.updateNegatives
  });

  trainer.initialize(tokenizer.vocabSize);

  // ============================================
  // SETUP SNAPSHOTS
  // ============================================

  if (CONFIG.saveSnapshotEvery > 0) {
    // Create snapshot directory
    if (!fs.existsSync(CONFIG.snapshotDir)) {
      fs.mkdirSync(CONFIG.snapshotDir, { recursive: true });
    }

    // Calculate which epochs will be saved
    const savedEpochs = [];
    for (let e = 0; e < CONFIG.epochs; e++) {
      if ((e + 1) % CONFIG.saveSnapshotEvery === 0) {
        savedEpochs.push(e + 1);
      }
    }
    console.log(`Snapshots will be saved at epoch(s): ${savedEpochs.join(', ')}`);
    console.log(`Snapshot directory: ${CONFIG.snapshotDir}\n`);
  } else {
    console.log('Snapshots disabled (saveSnapshotEvery = 0)\n');
  }

  // ============================================
  // TRAIN MODEL
  // ============================================

  const startTime = Date.now();

  trainer.train(corpus, (epoch, stats) => {
    // Save snapshot if configured
    if (CONFIG.saveSnapshotEvery > 0 && (epoch + 1) % CONFIG.saveSnapshotEvery === 0) {
      const snapshotPath = `${CONFIG.snapshotDir}/model_epoch${epoch + 1}.json`;
      console.log(`    💾 Saving snapshot: ${snapshotPath}`);
      ModelPersistence.saveModel(trainer, tokenizer, snapshotPath);
    }
  });

  const trainingTime = ((Date.now() - startTime) / 1000).toFixed(2);

  // ============================================
  // SAVE MODEL
  // ============================================

  console.log('\n' + '▓'.repeat(70));
  console.log('SAVING MODEL');
  console.log('▓'.repeat(70) + '\n');

  // Ensure model directory exists
  if (!fs.existsSync(CONFIG.modelDir)) {
    fs.mkdirSync(CONFIG.modelDir, { recursive: true });
  }

  // Generate model filename
  let modelPath = `${CONFIG.modelDir}/css_model.json`;
  if (CONFIG.useTimestampedModels) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    modelPath = `${CONFIG.modelDir}/css_model_${timestamp}.json`;
  }

  // Save model
  ModelPersistence.saveModel(trainer, tokenizer, modelPath);

  // Save training metadata
  const metadata = {
    timestamp: new Date().toISOString(),
    config: CONFIG,
    corpusInfo: {
      documents: corpusCache.corpusSize,
      vocabSize: corpusCache.vocabSize,
      source: corpusCache.config
    },
    trainingTime: `${trainingTime}s`
  };

  const metadataPath = modelPath.replace('.json', '.metadata.json');
  fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
  console.log(`✓ Model saved: ${modelPath}`);
  console.log(`✓ Metadata saved: ${metadataPath}`);

  // ============================================
  // TRAINING COMPLETE
  // ============================================

  console.log('\n' + '█'.repeat(70));
  console.log('TRAINING COMPLETE!');
  console.log('█'.repeat(70));

  console.log(`\nDocuments: ${corpusCache.corpusSize.toLocaleString()}`);
  console.log(`Vocabulary: ${corpusCache.vocabSize.toLocaleString()} words`);
  console.log(`Training time: ${trainingTime}s`);
  console.log(`Model: ${modelPath}`);
  console.log('\n' + '█'.repeat(70) + '\n');
}

// ============================================
// RUN TRAINING
// ============================================

train().catch(error => {
  console.error('\n❌ Training failed:');
  console.error(error);
  process.exit(1);
});
