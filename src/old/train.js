/**
 * Training Script - Train CSS model one Parquet file at a time
 *
 * Usage:
 *   npm run train
 *
 * Prerequisites:
 *   1. Download Parquet files: npm run download
 *   2. Install Python with pandas/pyarrow: pip install pandas pyarrow
 *
 * This script:
 * 1. Processes ONE parquet file at a time (memory efficient)
 * 2. Saves checkpoint after each parquet file
 * 3. Can resume from last checkpoint
 * 4. Supports parallel training (each parquet can be trained independently)
 */

import {Tokenizer} from './preprocessing/tokenizer.js';
import {CSSTrainer} from './core/CSSTrainer.js';
import {ModelPersistence} from './utils/ModelPersistence.js';
import fs from 'fs';
import path from 'path';
import {execSync} from 'child_process';

// ============================================
// CONFIGURATION
// ============================================

const CONFIG = {
    // Data Loading
    parquetDir: './data/parquet',
    maxDocLength: 5000,           // Max characters per document
    maxDocsPerFile: null,         // Max documents per file (null = all)

    // Vocabulary
    minWordFreq: 5,               // Minimum word frequency
    maxVocabSize: 50000,          // Maximum vocabulary size
    buildVocabFromAllFiles: true, // Build vocab from all files first (recommended)

    // Model Architecture
    frequencyDim: 200,            // Total frequency space
    maxFrequencies: 8,            // Max active frequencies per word
    windowSize: 3,                // Context window size

    // Training Hyperparameters
    learningRate: 0.03,           // Learning rate
    sparsityPenalty: 0.003,       // L1 sparsity penalty
    epochs: 10,                   // Number of training epochs per parquet
    batchSize: 100,               // Batch size for training

    // Contrastive Learning
    negativeCount: 5,             // Negative samples per positive
    margin: 0.5,                  // Contrastive margin
    updateNegatives: true,        // Update negative words

    // Checkpointing
    checkpointDir: './checkpoints',
    checkpointFile: 'training_checkpoint.json',

    // Output
    modelDir: './models',
    finalModelName: 'css_model_final.json',

    // Logging
    logEvery: 1000                // Log progress every N documents
};

// ============================================
// CHECKPOINT MANAGEMENT
// ============================================

function loadCheckpoint() {
    const checkpointPath = path.join(CONFIG.checkpointDir, CONFIG.checkpointFile);

    if (!fs.existsSync(checkpointPath)) {
        return null;
    }

    console.log('Found existing checkpoint, loading...\n');
    const checkpoint = ModelPersistence.loadCheckpoint(checkpointPath);
    return checkpoint;
}

function saveCheckpoint(checkpoint) {
    const checkpointPath = path.join(CONFIG.checkpointDir, CONFIG.checkpointFile);

    if (!fs.existsSync(CONFIG.checkpointDir)) {
        fs.mkdirSync(CONFIG.checkpointDir, {recursive: true});
    }

    ModelPersistence.saveCheckpoint(checkpoint, checkpointPath);
}

// ============================================
// VOCABULARY BUILDING
// ============================================

/**
 * Build vocabulary from ALL parquet files first
 * This ensures consistent tokenization across all batches
 */
async function buildGlobalVocabulary(files) {
    console.log('▓'.repeat(70));
    console.log('BUILDING GLOBAL VOCABULARY FROM ALL PARQUET FILES');
    console.log('▓'.repeat(70) + '\n');

    const tokenizer = new Tokenizer();

    for (let i = 0; i < files.length; i++) {
        const filename = files[i];
        const filepath = path.join(CONFIG.parquetDir, filename);

        console.log(`[${i + 1}/${files.length}] Scanning: ${filename}`);

        try {
            const texts = await loadParquetFile(filepath);

            console.log(`  Analyzing ${texts.length.toLocaleString()} documents...`);
            tokenizer.buildVocab(texts, CONFIG.minWordFreq);

            console.log(`  Current vocabulary: ${tokenizer.vocabSize.toLocaleString()} words`);

        } catch (error) {
            console.error(`\n❌ Error scanning ${filename}: ${error.message}`);
            throw error;
        }
    }

    console.log(`\n✓ Global vocabulary: ${tokenizer.vocabSize.toLocaleString()} words\n`);

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

// ============================================
// DATA LOADING
// ============================================

async function loadParquetFile(filepath) {
    const maxDocsArg = CONFIG.maxDocsPerFile ? ` ${CONFIG.maxDocsPerFile}` : '';
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

    return processedTexts;
}

function tokenizeTexts(texts, tokenizer) {
    const corpus = texts
        .map((text, idx) => {
            if (idx % CONFIG.logEvery === 0 && idx > 0) {
                process.stdout.write(`\r  Tokenizing: ${idx.toLocaleString()}/${texts.length.toLocaleString()}`);
            }
            return tokenizer.textToIds(text);
        })
        .filter(ids => ids.length > 0);

    if (texts.length > CONFIG.logEvery) {
        console.log(''); // New line after progress
    }

    return corpus;
}

// ============================================
// TRAINING
// ============================================

async function trainOnParquetFile(filename, fileIndex, totalFiles, trainer, tokenizer) {
    console.log('\n' + '▓'.repeat(70));
    console.log(`PROCESSING PARQUET ${fileIndex + 1}/${totalFiles}: ${filename}`);
    console.log('▓'.repeat(70) + '\n');

    const filepath = path.join(CONFIG.parquetDir, filename);
    const stats = fs.statSync(filepath);
    console.log(`File size: ${(stats.size / 1024 / 1024).toFixed(1)}MB\n`);

    // Load texts from this parquet file
    console.log('Loading documents...');
    const texts = await loadParquetFile(filepath);
    console.log(`✓ Loaded: ${texts.length.toLocaleString()} documents\n`);

    // Tokenize
    console.log('Tokenizing...');
    const corpus = tokenizeTexts(texts, tokenizer);
    console.log(`✓ Tokenized: ${corpus.length.toLocaleString()} documents\n`);

    // Free memory
    texts.length = 0;

    // Train on this batch
    console.log('Training...');
    trainer.train(corpus);

    // Free memory
    corpus.length = 0;

    console.log(`\n✓ Completed training on ${filename}`);
}

// ============================================
// MAIN TRAINING LOOP
// ============================================

async function main() {
    console.log('\n' + '█'.repeat(70));
    console.log('CSS MODEL TRAINING - INCREMENTAL PARQUET PROCESSING');
    console.log('█'.repeat(70) + '\n');

    const startTime = Date.now();

    try {
        // Check for parquet files
        if (!fs.existsSync(CONFIG.parquetDir)) {
            throw new Error(`Parquet directory not found: ${CONFIG.parquetDir}\n\nPlease run: npm run download`);
        }

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

        // Load or create checkpoint
        let checkpoint = loadCheckpoint();
        let tokenizer;
        let trainer;
        let startFileIndex = 0;

        if (checkpoint) {
            // Resume from checkpoint
            console.log('Resuming from checkpoint...');
            console.log(`  Last completed file: ${checkpoint.lastCompletedFile}`);
            console.log(`  Files processed: ${checkpoint.filesProcessed}/${files.length}`);
            console.log(`  Total documents trained: ${checkpoint.totalDocuments.toLocaleString()}\n`);

            startFileIndex = checkpoint.filesProcessed;

            // Reconstruct tokenizer
            tokenizer = new Tokenizer();
            tokenizer.vocab = new Map(Object.entries(checkpoint.vocab));
            tokenizer.idToWord = new Map(Object.entries(checkpoint.idToWord).map(([k, v]) => [parseInt(k), v]));
            tokenizer.wordFreq = new Map(Object.entries(checkpoint.wordFreq));
            tokenizer.nextId = tokenizer.vocab.size;

            // Reconstruct trainer
            trainer = new CSSTrainer(checkpoint.config);
            trainer.initialize(checkpoint.vocabSize);
            trainer.importModel({
                config: checkpoint.config,
                vocabSize: checkpoint.vocabSize,
                spectra: checkpoint.spectra
            });

            console.log('✓ Checkpoint restored\n');

        } else {
            // Start fresh
            console.log('Starting fresh training...\n');

            // Build global vocabulary
            tokenizer = await buildGlobalVocabulary(files);

            // Initialize trainer
            console.log('▓'.repeat(70));
            console.log('INITIALIZING CSS TRAINER');
            console.log('▓'.repeat(70) + '\n');

            trainer = new CSSTrainer({
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

            console.log('Configuration:');
            console.log(`  Frequency dimension: ${CONFIG.frequencyDim}`);
            console.log(`  Max frequencies/word: ${CONFIG.maxFrequencies}`);
            console.log(`  Window size: ${CONFIG.windowSize}`);
            console.log(`  Learning rate: ${CONFIG.learningRate}`);
            console.log(`  Sparsity penalty: ${CONFIG.sparsityPenalty}`);
            console.log(`  Epochs per batch: ${CONFIG.epochs}`);
            console.log(`  Negative samples: ${CONFIG.negativeCount}\n`);
        }

        // Process each parquet file
        let totalDocuments = checkpoint ? checkpoint.totalDocuments : 0;

        for (let i = startFileIndex; i < files.length; i++) {
            const filename = files[i];

            await trainOnParquetFile(filename, i, files.length, trainer, tokenizer);

            // Update totals (approximate - actual count would require tracking)
            totalDocuments += 10000; // Rough estimate

            // Save checkpoint
            console.log('\nSaving checkpoint...');
            checkpoint = {
                version: '1.0.0',
                timestamp: new Date().toISOString(),
                lastCompletedFile: filename,
                filesProcessed: i + 1,
                totalFiles: files.length,
                totalDocuments,
                config: trainer.config,
                vocabSize: trainer.vocabSize,
                vocab: Object.fromEntries(tokenizer.vocab),
                idToWord: Object.fromEntries(tokenizer.idToWord),
                wordFreq: Object.fromEntries(tokenizer.wordFreq),
                spectra: trainer.exportModel().spectra
            };

            saveCheckpoint(checkpoint);

            console.log(`\n${'='.repeat(70)}`);
            console.log(`PROGRESS: ${i + 1}/${files.length} files completed (${((i + 1) / files.length * 100).toFixed(1)}%)`);
            console.log(`${'='.repeat(70)}\n`);
        }

        // Save final model
        console.log('\n' + '▓'.repeat(70));
        console.log('SAVING FINAL MODEL');
        console.log('▓'.repeat(70) + '\n');

        if (!fs.existsSync(CONFIG.modelDir)) {
            fs.mkdirSync(CONFIG.modelDir, {recursive: true});
        }

        const finalModelPath = path.join(CONFIG.modelDir, CONFIG.finalModelName);
        ModelPersistence.saveModel(trainer, tokenizer, finalModelPath);

        // Complete
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

        console.log('\n' + '█'.repeat(70));
        console.log('TRAINING COMPLETE!');
        console.log('█'.repeat(70));
        console.log(`\nTime elapsed: ${elapsed}s`);
        console.log(`Model saved: ${finalModelPath}`);
        console.log(`Vocabulary: ${tokenizer.vocabSize.toLocaleString()} words`);
        console.log(`Parquet files processed: ${files.length}`);
        console.log(`Total documents: ~${totalDocuments.toLocaleString()}`);
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
