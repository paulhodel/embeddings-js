/**
 * Training Script - Train CSS model directly from Parquet files
 *
 * Usage:
 *   npm run train
 *
 * Prerequisites:
 *   1. Download Parquet files: npm run download
 *   2. Install Python with pandas/pyarrow: pip install pandas pyarrow
 *
 * This script:
 * 1. Reads Parquet files directly (no intermediate cache)
 * 2. Builds vocabulary in-memory
 * 3. Trains CSS model
 * 4. Saves model with timestamp
 * 5. Saves periodic snapshots
 */

import { Tokenizer } from './preprocessing/tokenizer.js';
import { CSSTrainer } from './core/CSSTrainer.js';
import { ModelPersistence } from './utils/ModelPersistence.js';
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

// ============================================
// CONFIGURATION
// ============================================

const CONFIG = {
  // Data Loading
  parquetDir: './data/parquet',
  maxDocLength: 5000,           // Max characters per document
  maxDocsPerFile: 10000,        // Max documents per file (null = all, 10000 = ~50k total)
  maxTotalDocs: 50000,          // Max total documents across all files (null = all)

  // Vocabulary
  minWordFreq: 5,               // Minimum word frequency
  maxVocabSize: 50000,          // Maximum vocabulary size

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
  snapshotDir: './models/snapshots',

  // Logging
  logEvery: 1000                // Log progress every N documents
};

// ============================================
// DATA LOADING FROM PARQUET
// ============================================

/**
 * Read all Parquet files and extract text
 */
async function loadParquetFiles() {
  console.log('▓'.repeat(70));
  console.log('LOADING DATA FROM PARQUET FILES');
  console.log('▓'.repeat(70) + '\n');

  if (!fs.existsSync(CONFIG.parquetDir)) {
    throw new Error(`Parquet directory not found: ${CONFIG.parquetDir}\n\nPlease run: npm run download`);
  }

  // Find all parquet files
  const files = fs.readdirSync(CONFIG.parquetDir)
    .filter(f => f.endsWith('.parquet'))
    .sort();

  if (files.length === 0) {
    throw new Error(`No Parquet files found in: ${CONFIG.parquetDir}\n\nPlease run: npm run download`);
  }

  console.log(`Found ${files.length} Parquet file(s):\n`);
  files.forEach((f, i) => {
    const stats = fs.statSync(path.join(CONFIG.parquetDir, f));
    console.log(`  ${i + 1}. ${f} (${(stats.size / 1024 / 1024).toFixed(1)}MB)`);
  });
  console.log('');

  let allTexts = [];
  let totalDocs = 0;

  for (let i = 0; i < files.length; i++) {
    // Check if we've hit the total document limit
    if (CONFIG.maxTotalDocs && totalDocs >= CONFIG.maxTotalDocs) {
      console.log(`\n✓ Reached max total documents limit: ${CONFIG.maxTotalDocs.toLocaleString()}`);
      break;
    }

    const filename = files[i];
    const filepath = path.join(CONFIG.parquetDir, filename);

    console.log(`[${i + 1}/${files.length}] Reading: ${filename}`);

    try {
      // Calculate how many docs we can still load
      const remainingDocs = CONFIG.maxTotalDocs ? CONFIG.maxTotalDocs - totalDocs : null;
      const docsToLoad = CONFIG.maxDocsPerFile && remainingDocs
        ? Math.min(CONFIG.maxDocsPerFile, remainingDocs)
        : (CONFIG.maxDocsPerFile || remainingDocs);

      // Use Python to read Parquet file
      const maxDocsArg = docsToLoad ? ` ${docsToLoad}` : '';
      const pythonCmd = `python scripts/read_parquet.py "${filepath}"${maxDocsArg}`;

      const output = execSync(pythonCmd, {
        encoding: 'utf8',
        maxBuffer: 500 * 1024 * 1024 // 500MB buffer
      });

      const texts = JSON.parse(output);

      // Truncate long documents
      const processedTexts = texts.map(text =>
        text.length > CONFIG.maxDocLength ? text.substring(0, CONFIG.maxDocLength) : text
      );

      allTexts.push(...processedTexts);
      totalDocs += processedTexts.length;

      console.log(`  ✓ Loaded: ${processedTexts.length.toLocaleString()} documents (total: ${totalDocs.toLocaleString()})`);

    } catch (error) {
      console.error(`\n❌ Error reading ${filename}: ${error.message}`);
      if (error.stderr) {
        console.error(`   stderr: ${error.stderr.toString()}`);
      }
      throw error;
    }
  }

  console.log(`\n✓ Total documents loaded: ${totalDocs.toLocaleString()}\n`);

  return allTexts;
}

// ============================================
// BUILD VOCABULARY AND TOKENIZE
// ============================================

function buildVocabulary(texts) {
  console.log('▓'.repeat(70));
  console.log('BUILDING VOCABULARY');
  console.log('▓'.repeat(70) + '\n');

  const tokenizer = new Tokenizer();

  console.log('Analyzing word frequencies...');
  tokenizer.buildVocab(texts, CONFIG.minWordFreq);

  console.log(`✓ Initial vocabulary: ${tokenizer.vocabSize.toLocaleString()} words\n`);

  // Limit vocabulary size if needed
  if (tokenizer.vocabSize > CONFIG.maxVocabSize) {
    console.log(`Limiting vocabulary to top ${CONFIG.maxVocabSize.toLocaleString()} words...`);

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

    console.log(`✓ Limited to: ${tokenizer.vocabSize.toLocaleString()} words\n`);
  }

  return tokenizer;
}

function tokenizeCorpus(texts, tokenizer) {
  console.log('▓'.repeat(70));
  console.log('TOKENIZING CORPUS');
  console.log('▓'.repeat(70) + '\n');

  console.log('Converting texts to word ID sequences...');
  const corpus = texts
    .map((text, idx) => {
      if (idx % CONFIG.logEvery === 0 && idx > 0) {
        process.stdout.write(`\r  Progress: ${idx.toLocaleString()}/${texts.length.toLocaleString()}`);
      }
      return tokenizer.textToIds(text);
    })
    .filter(ids => ids.length > 0);

  console.log(`\n✓ Tokenized: ${corpus.length.toLocaleString()} documents\n`);

  return corpus;
}

// ============================================
// TRAINING
// ============================================

async function trainModel(corpus, tokenizer) {
  console.log('▓'.repeat(70));
  console.log('TRAINING CSS MODEL');
  console.log('▓'.repeat(70) + '\n');

  console.log('Configuration:');
  console.log(`  Frequency dimension: ${CONFIG.frequencyDim}`);
  console.log(`  Max frequencies/word: ${CONFIG.maxFrequencies}`);
  console.log(`  Window size: ${CONFIG.windowSize}`);
  console.log(`  Learning rate: ${CONFIG.learningRate}`);
  console.log(`  Sparsity penalty: ${CONFIG.sparsityPenalty}`);
  console.log(`  Epochs: ${CONFIG.epochs}`);
  console.log(`  Negative samples: ${CONFIG.negativeCount}`);
  console.log(`  Margin: ${CONFIG.margin}`);
  console.log('');

  // Initialize trainer
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

  // Train
  console.log('Starting training...\n');
  trainer.train(corpus);

  return trainer;
}

// ============================================
// SAVE MODEL
// ============================================

function saveModel(trainer, tokenizer) {
  console.log('\n' + '▓'.repeat(70));
  console.log('SAVING MODEL');
  console.log('▓'.repeat(70) + '\n');

  // Create model directory
  if (!fs.existsSync(CONFIG.modelDir)) {
    fs.mkdirSync(CONFIG.modelDir, { recursive: true });
  }

  // Generate filename
  const timestamp = CONFIG.useTimestampedModels
    ? new Date().toISOString().replace(/[:.]/g, '-').split('T')[0]
    : '';

  const filename = CONFIG.useTimestampedModels
    ? `css_model_${timestamp}.json`
    : 'css_model.json';

  const filepath = path.join(CONFIG.modelDir, filename);

  // Save
  ModelPersistence.saveModel(trainer, tokenizer, filepath);

  return filepath;
}

// ============================================
// MAIN
// ============================================

async function main() {
  console.log('\n' + '█'.repeat(70));
  console.log('CSS MODEL TRAINING');
  console.log('█'.repeat(70) + '\n');

  const startTime = Date.now();

  try {
    // Step 1: Load data from Parquet files
    const texts = await loadParquetFiles();

    // Step 2: Build vocabulary
    const tokenizer = buildVocabulary(texts);

    // Step 3: Tokenize corpus
    const corpus = tokenizeCorpus(texts, tokenizer);

    // Free memory
    texts.length = 0;

    // Step 4: Train model
    const trainer = await trainModel(corpus, tokenizer);

    // Step 5: Save model
    const modelPath = saveModel(trainer, tokenizer);

    // Complete
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    console.log('\n' + '█'.repeat(70));
    console.log('TRAINING COMPLETE!');
    console.log('█'.repeat(70));
    console.log(`\nTime elapsed: ${elapsed}s`);
    console.log(`Model saved: ${modelPath}`);
    console.log(`Vocabulary: ${tokenizer.vocabSize.toLocaleString()} words`);
    console.log(`Documents: ${corpus.length.toLocaleString()}`);
    console.log('\n' + '█'.repeat(70) + '\n');

  } catch (error) {
    console.error('\n❌ Training failed:');
    console.error(error.message);
    if (error.stack) {
      console.error('\nStack trace:');
      console.error(error.stack);
    }
    process.exit(1);
  }
}

main();
