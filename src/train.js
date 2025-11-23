/**
 * Training script for CSS model using HuggingFace dataset streaming
 *
 * Usage:
 *   npm run train
 *
 * Features:
 * - Streams data from HuggingFace fineweb-edu-100b-shuffle dataset
 * - Real-time progress tracking
 * - Automatic checkpointing (can pause and resume)
 * - Model persistence
 */

import { Tokenizer } from './preprocessing/tokenizer.js';
import { CSSTrainer } from './core/CSSTrainer.js';
import { HuggingFaceStreamer } from './data/HuggingFaceStreamer.js';
import { ModelPersistence } from './utils/ModelPersistence.js';
import fs from 'fs';

// ============================================
// CONFIGURATION
// ============================================

const CONFIG = {
  // Dataset
  dataset: 'karpathy/fineweb-edu-100b-shuffle',
  split: 'train',
  maxBatches: 100,              // Set to null for unlimited (full dataset)
  batchSize: 50,                // Documents per fetch
  maxDocLength: 5000,           // Max characters per document

  // Vocabulary
  minWordFreq: 5,               // Minimum word frequency
  maxVocabSize: 50000,          // Maximum vocabulary size

  // Model
  frequencyDim: 200,            // Total frequency space
  maxFrequencies: 8,            // Max active frequencies per word
  windowSize: 3,                // Context window size
  learningRate: 0.03,           // Learning rate
  sparsityPenalty: 0.003,       // L1 sparsity penalty
  negativeCount: 5,             // Negative samples per positive
  margin: 0.5,                  // Contrastive margin
  updateNegatives: true,        // Update negative words

  // Training
  epochs: 5,                    // Number of training epochs
  trainingBatchSize: 100,       // Batch size for training

  // Checkpointing
  saveCheckpointEvery: 20,      // Save checkpoint every N batches during data collection
  resumeFromCheckpoint: true,   // Try to resume from checkpoint if exists
  checkpointPath: './checkpoints/training_checkpoint.json',
  corpusCachePath: './checkpoints/corpus_cache.json', // Cache corpus after streaming
  modelPath: './models/css_model.json',
  useTimestampedModels: true,   // Add timestamp to model filenames

  // Model Snapshots (for stability analysis)
  saveSnapshotEvery: 1,         // Save model snapshot every N epochs (1=every epoch, 2=every other epoch, etc.)
  snapshotDir: './models/snapshots',  // Set to 0 to disable snapshots

  // Logging
  logEvery: 5                   // Log progress every N batches
};

// ============================================
// TRAINING STATE
// ============================================

class TrainingState {
  constructor() {
    this.phase = 'streaming';     // 'streaming', 'training', 'complete'
    this.batchesProcessed = 0;
    this.documentsProcessed = 0;
    this.tokensProcessed = 0;
    this.startTime = Date.now();
    this.corpusTexts = [];
  }

  update(numDocs, numTokens) {
    this.batchesProcessed++;
    this.documentsProcessed += numDocs;
    this.tokensProcessed += numTokens;
  }

  getStats() {
    const elapsedSeconds = (Date.now() - this.startTime) / 1000;
    const docsPerSecond = this.documentsProcessed / elapsedSeconds;
    const tokensPerSecond = this.tokensProcessed / elapsedSeconds;

    return {
      batches: this.batchesProcessed,
      documents: this.documentsProcessed,
      tokens: this.tokensProcessed,
      elapsed: elapsedSeconds.toFixed(1),
      docsPerSec: docsPerSecond.toFixed(1),
      tokensPerSec: tokensPerSecond.toFixed(0)
    };
  }

  printProgress(streamer) {
    const stats = this.getStats();
    const progress = streamer.getProgress();

    console.log('\n' + '='.repeat(70));
    console.log('TRAINING PROGRESS');
    console.log('='.repeat(70));
    console.log(`Phase: ${this.phase.toUpperCase()}`);
    console.log(`Batches processed: ${stats.batches}`);
    console.log(`Documents: ${stats.documents.toLocaleString()}`);
    console.log(`Tokens: ${stats.tokens.toLocaleString()}`);
    console.log(`Elapsed time: ${stats.elapsed}s`);
    console.log(`Speed: ${stats.docsPerSec} docs/s, ${stats.tokensPerSec} tokens/s`);

    if (progress.total) {
      console.log(`Dataset progress: ${progress.processed.toLocaleString()}/${progress.total.toLocaleString()} (${progress.percentage}%)`);
    } else {
      console.log(`Dataset rows processed: ${progress.processed.toLocaleString()}`);
    }

    console.log('='.repeat(70) + '\n');
  }
}

// ============================================
// CHECKPOINT MANAGEMENT
// ============================================

function saveCheckpoint(state, tokenizer, config) {
  const checkpoint = {
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    phase: state.phase,
    batchesProcessed: state.batchesProcessed,
    documentsProcessed: state.documentsProcessed,
    tokensProcessed: state.tokensProcessed,
    corpusTexts: state.corpusTexts,
    vocab: Object.fromEntries(tokenizer.vocab),
    wordFreq: Object.fromEntries(tokenizer.wordFreq),
    config: config
  };

  ModelPersistence.saveCheckpoint(checkpoint, config.checkpointPath);
}

function loadCheckpoint(config) {
  const checkpoint = ModelPersistence.loadCheckpoint(config.checkpointPath);

  if (!checkpoint) {
    return null;
  }

  console.log('\n' + '▶'.repeat(70));
  console.log('RESUMING FROM CHECKPOINT');
  console.log('▶'.repeat(70));
  console.log(`Checkpoint from: ${checkpoint.timestamp}`);
  console.log(`Phase: ${checkpoint.phase}`);
  console.log(`Batches processed: ${checkpoint.batchesProcessed}`);
  console.log(`Documents: ${checkpoint.documentsProcessed.toLocaleString()}`);
  console.log(`Tokens: ${checkpoint.tokensProcessed.toLocaleString()}`);
  console.log('▶'.repeat(70) + '\n');

  return checkpoint;
}

function saveCorpusCache(corpus, tokenizer, config) {
  const cache = {
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    corpus: corpus,
    vocab: Object.fromEntries(tokenizer.vocab),
    wordFreq: Object.fromEntries(tokenizer.wordFreq),
    corpusSize: corpus.length,
    vocabSize: tokenizer.vocabSize
  };

  ModelPersistence.saveCheckpoint(cache, config.corpusCachePath);
  console.log(`✓ Corpus cached to: ${config.corpusCachePath}`);
}

function loadCorpusCache(config) {
  if (!fs.existsSync(config.corpusCachePath)) {
    return null;
  }

  try {
    const cache = ModelPersistence.loadCheckpoint(config.corpusCachePath);
    if (cache && cache.corpus && cache.vocab) {
      console.log('\n' + '▶'.repeat(70));
      console.log('LOADING CACHED CORPUS');
      console.log('▶'.repeat(70));
      console.log(`Cache from: ${cache.timestamp}`);
      console.log(`Documents: ${cache.corpusSize.toLocaleString()}`);
      console.log(`Vocabulary: ${cache.vocabSize.toLocaleString()} words`);
      console.log('▶'.repeat(70) + '\n');
      return cache;
    }
  } catch (error) {
    console.warn(`Could not load corpus cache: ${error.message}`);
  }

  return null;
}

// ============================================
// MAIN TRAINING FUNCTION
// ============================================

async function train() {
  console.log('\n' + '█'.repeat(70));
  console.log('COMPRESSIVE SEMANTIC SPECTROSCOPY - LARGE SCALE TRAINING');
  console.log('█'.repeat(70) + '\n');

  console.log('Configuration:');
  console.log(`  Dataset: ${CONFIG.dataset}`);
  console.log(`  Max batches: ${CONFIG.maxBatches || 'unlimited'}`);
  console.log(`  Batch size: ${CONFIG.batchSize}`);
  console.log(`  Frequency dim: ${CONFIG.frequencyDim}`);
  console.log(`  Max frequencies: ${CONFIG.maxFrequencies}`);
  console.log(`  Vocabulary size: up to ${CONFIG.maxVocabSize.toLocaleString()}`);
  console.log(`  Checkpoint: ${CONFIG.resumeFromCheckpoint ? 'enabled' : 'disabled'}`);
  console.log('');

  // Try to load checkpoint
  let checkpoint = null;
  if (CONFIG.resumeFromCheckpoint) {
    checkpoint = loadCheckpoint(CONFIG);
  }

  // Initialize components
  const tokenizer = new Tokenizer();
  const state = new TrainingState();

  // ============================================
  // PHASE 1: STREAM DATA & BUILD VOCABULARY
  // ============================================

  let corpus = [];
  let skipStreaming = false;
  let allTexts = [];

  // First, try to load corpus cache (avoids re-fetching from HuggingFace)
  const corpusCache = loadCorpusCache(CONFIG);
  if (corpusCache) {
    console.log('Using cached corpus (skipping HuggingFace streaming)...\n');
    corpus = corpusCache.corpus;

    // Restore tokenizer
    tokenizer.vocab = new Map(Object.entries(corpusCache.vocab));
    tokenizer.wordFreq = new Map(Object.entries(corpusCache.wordFreq));
    tokenizer.nextId = tokenizer.vocab.size;

    for (const [word, id] of tokenizer.vocab.entries()) {
      tokenizer.idToWord.set(id, word);
    }

    skipStreaming = true;
  } else if (checkpoint && checkpoint.phase === 'streaming' && checkpoint.corpusTexts) {
    // Resume from checkpoint
    console.log('Resuming data streaming from checkpoint...\n');
    const allTexts = checkpoint.corpusTexts;
    state.batchesProcessed = checkpoint.batchesProcessed;
    state.documentsProcessed = checkpoint.documentsProcessed;
    state.tokensProcessed = checkpoint.tokensProcessed;
    state.corpusTexts = allTexts;

    // Restore tokenizer state
    tokenizer.vocab = new Map(Object.entries(checkpoint.vocab));
    tokenizer.wordFreq = new Map(Object.entries(checkpoint.wordFreq));
    tokenizer.nextId = tokenizer.vocab.size;

    // Rebuild idToWord
    for (const [word, id] of tokenizer.vocab.entries()) {
      tokenizer.idToWord.set(id, word);
    }

    console.log(`✓ Restored ${allTexts.length.toLocaleString()} documents from checkpoint`);
    console.log(`✓ Restored vocabulary: ${tokenizer.vocabSize.toLocaleString()} words\n`);

    // Convert to corpus now (we'll continue below)
    skipStreaming = true;
    console.log('Converting texts to word ID sequences...');
    corpus = allTexts
      .map(text => tokenizer.textToIds(text))
      .filter(ids => ids.length > 0);
    console.log(`✓ Corpus created: ${corpus.length.toLocaleString()} documents\n`);
  }

  if (!skipStreaming) {
    // Start fresh
    console.log('\n' + '▓'.repeat(70));
    console.log('PHASE 1: STREAMING DATA & BUILDING VOCABULARY');
    console.log('▓'.repeat(70) + '\n');

    const streamer = new HuggingFaceStreamer(CONFIG.dataset, CONFIG.split, CONFIG.batchSize);
    await streamer.initialize();

    console.log('Streaming data from HuggingFace...\n');

    for await (const batch of streamer.stream(CONFIG.maxBatches)) {
      state.batchesProcessed++;

      // Truncate long documents
      const processedBatch = batch.map(text =>
        text.length > CONFIG.maxDocLength ? text.substring(0, CONFIG.maxDocLength) : text
      );

      allTexts.push(...processedBatch);
      state.corpusTexts = allTexts;

      // Count tokens
      const numTokens = processedBatch.reduce((sum, text) => {
        return sum + tokenizer.tokenize(text).length;
      }, 0);

      state.update(processedBatch.length, numTokens);

      // Log progress
      if (state.batchesProcessed % CONFIG.logEvery === 0) {
        state.printProgress(streamer);
      }

      // Save checkpoint
      if (state.batchesProcessed % CONFIG.saveCheckpointEvery === 0) {
        console.log('💾 Saving checkpoint...');
        saveCheckpoint(state, tokenizer, CONFIG);
      }

      // Memory management: build vocab in chunks
      if (allTexts.length >= 10000 && allTexts.length % 10000 === 0) {
        console.log(`Building vocabulary from ${allTexts.length.toLocaleString()} documents...`);
        tokenizer.buildVocab(allTexts, CONFIG.minWordFreq);
        console.log(`Current vocabulary size: ${tokenizer.vocabSize.toLocaleString()}\n`);
      }
    }
  }

  // Only build vocabulary if we streamed data
  if (!skipStreaming) {
    // Final vocabulary build
    console.log('\n' + '▓'.repeat(70));
    console.log('FINALIZING VOCABULARY');
    console.log('▓'.repeat(70) + '\n');

    tokenizer.buildVocab(allTexts, CONFIG.minWordFreq);

    // Limit vocabulary size if needed
    if (tokenizer.vocabSize > CONFIG.maxVocabSize) {
      console.log(`Limiting vocabulary to top ${CONFIG.maxVocabSize.toLocaleString()} words by frequency...`);
      const sortedWords = Array.from(tokenizer.wordFreq.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, CONFIG.maxVocabSize);

      tokenizer.vocab.clear();
      tokenizer.idToWord.clear();
      tokenizer.nextId = 0;

      for (const [word, freq] of sortedWords) {
        tokenizer.vocab.set(word, tokenizer.nextId);
        tokenizer.idToWord.set(tokenizer.nextId, word);
        tokenizer.nextId++;
      }

      console.log(`✓ Vocabulary limited to ${tokenizer.vocabSize.toLocaleString()} words\n`);
    }

    // ============================================
    // PHASE 2: CONVERT TO CORPUS
    // ============================================

    console.log('\n' + '▓'.repeat(70));
    console.log('PHASE 2: CONVERTING TO CORPUS');
    console.log('▓'.repeat(70) + '\n');

    console.log('Converting texts to word ID sequences...');
    corpus = allTexts
      .map(text => tokenizer.textToIds(text))
      .filter(ids => ids.length > 0);

    console.log(`✓ Corpus created: ${corpus.length.toLocaleString()} documents\n`);

    // Save corpus cache for future runs
    console.log('Saving corpus cache...');
    saveCorpusCache(corpus, tokenizer, CONFIG);

    // Clear text data to free memory
    allTexts = null;
    state.corpusTexts = null;
  }

  // ============================================
  // PHASE 3: TRAIN CSS MODEL
  // ============================================

  state.phase = 'training';
  console.log('\n' + '▓'.repeat(70));
  console.log('PHASE 3: TRAINING CSS MODEL');
  console.log('▓'.repeat(70) + '\n');

  const trainer = new CSSTrainer({
    frequencyDim: CONFIG.frequencyDim,
    maxFrequencies: CONFIG.maxFrequencies,
    windowSize: CONFIG.windowSize,
    learningRate: CONFIG.learningRate,
    sparsityPenalty: CONFIG.sparsityPenalty,
    epochs: CONFIG.epochs,
    batchSize: CONFIG.trainingBatchSize,
    negativeCount: CONFIG.negativeCount,
    margin: CONFIG.margin,
    updateNegatives: CONFIG.updateNegatives
  });

  trainer.initialize(tokenizer.vocabSize);

  // Create snapshot directory if needed
  if (CONFIG.saveSnapshotEvery > 0 && !fs.existsSync(CONFIG.snapshotDir)) {
    fs.mkdirSync(CONFIG.snapshotDir, { recursive: true });
  }

  // Train with periodic snapshots
  if (CONFIG.saveSnapshotEvery > 0) {
    // Calculate which epochs will be saved
    const savedEpochs = [];
    for (let e = 0; e < CONFIG.epochs; e++) {
      if ((e + 1) % CONFIG.saveSnapshotEvery === 0) {
        savedEpochs.push(e + 1);
      }
    }
    console.log(`Model snapshots will be saved at epoch(s): ${savedEpochs.join(', ')}`);
    console.log(`Snapshot directory: ${CONFIG.snapshotDir}\n`);
  } else {
    console.log('Model snapshots disabled (saveSnapshotEvery = 0)\n');
  }

  // Train model with epoch callback for snapshot saving
  // Note: trainer.train() now includes two-phase pruning and normalization
  trainer.train(corpus, (epoch, stats) => {
    // Save snapshot if configured
    if (CONFIG.saveSnapshotEvery > 0 && (epoch + 1) % CONFIG.saveSnapshotEvery === 0) {
      const snapshotPath = `${CONFIG.snapshotDir}/model_epoch${epoch + 1}.json`;
      console.log(`    💾 Saving snapshot: ${snapshotPath}`);
      ModelPersistence.saveModel(trainer, tokenizer, snapshotPath);
    }
  });

  // ============================================
  // PHASE 4: SAVE MODEL
  // ============================================

  state.phase = 'complete';
  console.log('\n' + '▓'.repeat(70));
  console.log('PHASE 4: SAVING MODEL');
  console.log('▓'.repeat(70) + '\n');

  // Generate model filename (with or without timestamp)
  let finalModelPath = CONFIG.modelPath;
  if (CONFIG.useTimestampedModels) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const baseDir = CONFIG.modelPath.substring(0, CONFIG.modelPath.lastIndexOf('/'));
    const baseName = CONFIG.modelPath.substring(CONFIG.modelPath.lastIndexOf('/') + 1, CONFIG.modelPath.lastIndexOf('.json'));
    finalModelPath = `${baseDir}/${baseName}_${timestamp}.json`;
  } else {
    // Backup existing model if present
    if (fs.existsSync(CONFIG.modelPath)) {
      ModelPersistence.backupModel(CONFIG.modelPath);
    }
  }

  // Save model
  ModelPersistence.saveModel(trainer, tokenizer, finalModelPath);

  // Save training metadata
  const metadata = {
    timestamp: new Date().toISOString(),
    config: CONFIG,
    trainingStats: state.getStats()
  };

  const metadataPath = finalModelPath.replace('.json', '.metadata.json');
  fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
  console.log(`✓ Metadata saved: ${metadataPath}`);

  // Keep streaming checkpoint for corpus cache, but delete training checkpoint
  if (fs.existsSync(CONFIG.checkpointPath)) {
    fs.unlinkSync(CONFIG.checkpointPath);
    console.log(`✓ Streaming checkpoint deleted (training complete)`);
  }

  console.log(`\n💡 Tip: Corpus is cached at ${CONFIG.corpusCachePath}`);
  console.log(`   Future training runs will skip HuggingFace streaming and use the cache.`);
  console.log(`   Delete the cache file to re-fetch data from HuggingFace.`);

  // ============================================
  // TRAINING COMPLETE
  // ============================================

  console.log('\n' + '█'.repeat(70));
  console.log('TRAINING COMPLETE!');
  console.log('█'.repeat(70));

  const finalStats = state.getStats();
  console.log(`\nTotal documents: ${finalStats.documents.toLocaleString()}`);
  console.log(`Total tokens: ${finalStats.tokens.toLocaleString()}`);
  console.log(`Vocabulary size: ${tokenizer.vocabSize.toLocaleString()}`);
  console.log(`Training time: ${finalStats.elapsed}s`);
  console.log(`\nModel saved to: ${finalModelPath}`);
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
