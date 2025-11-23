/**
 * CSS (Compressive Semantic Spectroscopy) Training Script
 *
 * This implements the full CSS algorithm:
 * 1. Words as sparse complex spectra (amplitude + phase at specific frequencies)
 * 2. Random initialization with tiny amplitudes
 * 3. Contrastive learning with negative sampling
 * 4. Two-phase pruning (exploration + refinement)
 * 5. Amplitude normalization
 */

import fs from 'fs';
import path from 'path';
import {execSync} from 'child_process';
import readline from 'readline';
import * as vocabulary from './vocabulary.js';

// ============================================
// CONFIGURATION
// ============================================

const CONFIG = {
    // Data
    parquetDir: './data/parquet',
    stateFile: './data/training_state.json',
    vocabularyFile: './data/vocabulary.json',

    // CSS Model Architecture
    frequencyDim: 256,              // Total semantic frequency space (N)
    minInitFrequencies: 4,          // Start sparse: random 4-8 frequencies per word
    maxInitFrequencies: 8,
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
    snapshotDir: './data/snapshots', // Directory for model snapshots
};

// ============================================
// SPECTRUM UTILITIES
// ============================================

/**
 * Initialize a word with random sparse spectrum (variable size 4-8)
 * Returns: {frequencies: [...], amplitudes: [...], phases: [...]}
 */
function initializeSpectrum() {
    // Random number of frequencies between min and max
    const numFreqs = CONFIG.minInitFrequencies +
        Math.floor(Math.random() * (CONFIG.maxInitFrequencies - CONFIG.minInitFrequencies + 1));

    const frequencies = [];
    const amplitudes = [];
    const phases = [];

    // Choose K random unique frequency indices
    const usedFreqs = new Set();
    while (frequencies.length < numFreqs) {
        const freq = Math.floor(Math.random() * CONFIG.frequencyDim);
        if (!usedFreqs.has(freq)) {
            usedFreqs.add(freq);
            frequencies.push(freq);

            // TINY random amplitude [0.0005, 0.003]
            const amp = Math.random() * (CONFIG.initAmpMax - CONFIG.initAmpMin) + CONFIG.initAmpMin;
            amplitudes.push(amp);

            // Random phase [0, 2π]
            phases.push(Math.random() * 2 * Math.PI);
        }
    }

    return { frequencies, amplitudes, phases };
}

/**
 * Convert sparse spectrum to dense complex vector [real1, imag1, real2, imag2, ...]
 */
function spectrumToDense(spectrum) {
    const dense = new Float32Array(CONFIG.frequencyDim * 2); // real + imaginary

    for (let i = 0; i < spectrum.frequencies.length; i++) {
        const freq = spectrum.frequencies[i];
        const amp = spectrum.amplitudes[i];
        const phase = spectrum.phases[i];

        dense[freq * 2] = amp * Math.cos(phase);      // Real part
        dense[freq * 2 + 1] = amp * Math.sin(phase);  // Imaginary part
    }

    return dense;
}

/**
 * Build context spectrum by summing context word spectra
 */
function buildContextSpectrum(contextWordObjs) {
    const contextDense = new Float32Array(CONFIG.frequencyDim * 2);

    for (const wordObj of contextWordObjs) {
        if (!wordObj.spectrum || wordObj.spectrum.length === 0) {
            // Word not initialized yet, initialize it
            wordObj.spectrum = initializeSpectrum();
        }

        const wordDense = spectrumToDense(wordObj.spectrum);
        for (let i = 0; i < contextDense.length; i++) {
            contextDense[i] += wordDense[i];
        }
    }

    // Normalize by context size
    if (contextWordObjs.length > 0) {
        for (let i = 0; i < contextDense.length; i++) {
            contextDense[i] /= contextWordObjs.length;
        }
    }

    return contextDense;
}

/**
 * Compute compatibility score between word and context
 * Returns: real part of complex inner product
 */
function computeScore(wordSpectrum, contextDense) {
    const wordDense = spectrumToDense(wordSpectrum);

    let realPart = 0;
    let imagPart = 0;

    for (let i = 0; i < CONFIG.frequencyDim; i++) {
        const wReal = wordDense[i * 2];
        const wImag = wordDense[i * 2 + 1];
        const cReal = contextDense[i * 2];
        const cImag = contextDense[i * 2 + 1];

        // Complex conjugate inner product
        realPart += wReal * cReal + wImag * cImag;
        imagPart += wImag * cReal - wReal * cImag;
    }

    return Math.sqrt(realPart * realPart + imagPart * imagPart);
}

/**
 * Update spectrum with gradient descent + sparsity penalty
 */
function updateSpectrum(spectrum, gradient, learningRate, sparsityPenalty) {
    for (let i = 0; i < spectrum.frequencies.length; i++) {
        const freq = spectrum.frequencies[i];
        const amp = spectrum.amplitudes[i];
        const phase = spectrum.phases[i];

        // Current complex value
        const real = amp * Math.cos(phase);
        const imag = amp * Math.sin(phase);

        // Gradient at this frequency
        const gradReal = gradient[freq * 2];
        const gradImag = gradient[freq * 2 + 1];

        // Gradient descent
        const newReal = real - learningRate * gradReal;
        const newImag = imag - learningRate * gradImag;

        // Convert back to amplitude/phase
        let newAmp = Math.sqrt(newReal * newReal + newImag * newImag);
        const newPhase = Math.atan2(newImag, newReal);

        // Apply sparsity penalty (soft thresholding)
        newAmp = Math.max(0, newAmp - sparsityPenalty);

        spectrum.amplitudes[i] = newAmp;
        spectrum.phases[i] = newPhase;
    }

    // Normalize amplitudes (L2 norm)
    const norm = Math.sqrt(spectrum.amplitudes.reduce((sum, amp) => sum + amp * amp, 0));
    if (norm > 1e-10) {
        for (let i = 0; i < spectrum.amplitudes.length; i++) {
            spectrum.amplitudes[i] /= norm;
        }
    }
}

/**
 * Prune frequencies below threshold
 */
function pruneSpectrum(spectrum, threshold) {
    const keep = [];
    const keepAmps = [];
    const keepPhases = [];

    for (let i = 0; i < spectrum.amplitudes.length; i++) {
        if (spectrum.amplitudes[i] > threshold) {
            keep.push(spectrum.frequencies[i]);
            keepAmps.push(spectrum.amplitudes[i]);
            keepPhases.push(spectrum.phases[i]);
        }
    }

    spectrum.frequencies = keep;
    spectrum.amplitudes = keepAmps;
    spectrum.phases = keepPhases;
}

// ============================================
// TRAINING
// ============================================

/**
 * Process a single document with CSS training
 */
function processDocument(content, state) {
    // Sliding window context
    const context = [];
    const centerIndex = CONFIG.windowSize;
    const totalWindowSize = CONFIG.windowSize * 2 + 1;

    let token = '';

    for (let i = 0; i < content.length; i++) {
        const char = content[i];
        const code = char.charCodeAt(0);

        const isAlpha = (code >= 65 && code <= 90) || (code >= 97 && code <= 122);
        const isDigit = (code >= 48 && code <= 57);

        if (isAlpha || isDigit) {
            token += isAlpha && code <= 90 ? char.toLowerCase() : char;
        } else {
            if (token.length > 0) {
                // Get or create word object
                const wordObj = vocabulary.addWord(token);

                // Initialize spectrum if needed
                if (!wordObj.spectrum || wordObj.spectrum.length === 0) {
                    wordObj.spectrum = initializeSpectrum();
                }

                context.push(wordObj);

                // If window is full, train
                if (context.length === totalWindowSize) {
                    trainWindow(context, centerIndex, state);
                    context.shift();
                }

                token = '';
            }
        }
    }

    // Handle last token
    if (token.length > 0) {
        const wordObj = vocabulary.addWord(token);
        if (!wordObj.spectrum || wordObj.spectrum.length === 0) {
            wordObj.spectrum = initializeSpectrum();
        }
        context.push(wordObj);
        if (context.length === totalWindowSize) {
            trainWindow(context, centerIndex, state);
        }
    }
}

/**
 * Train on a single window using contrastive learning
 */
function trainWindow(context, centerIndex, state) {
    const centerWord = context[centerIndex];
    const contextWords = context.filter((_, i) => i !== centerIndex);

    // Build context spectrum
    const contextSpectrum = buildContextSpectrum(contextWords);

    // Positive score
    const posScore = computeScore(centerWord.spectrum, contextSpectrum);

    // Sample negative words
    const vocabSize = vocabulary.getVocabSize();
    const negativeWords = [];

    while (negativeWords.length < CONFIG.negativeCount) {
        const randomId = Math.floor(Math.random() * vocabSize);
        const negWord = vocabulary.getWordById(randomId);

        if (negWord && negWord.id !== centerWord.id) {
            if (!negWord.spectrum || negWord.spectrum.length === 0) {
                negWord.spectrum = initializeSpectrum();
            }
            negativeWords.push(negWord);
            break;
        }
    }

    // Compute gradients
    let hasLoss = false;
    const posGradient = new Float32Array(CONFIG.frequencyDim * 2);

    for (const negWord of negativeWords) {
        const negScore = computeScore(negWord.spectrum, contextSpectrum);
        const loss = CONFIG.margin - posScore + negScore;

        if (loss > 0) {
            hasLoss = true;

            // Positive gradient: push toward context
            for (let i = 0; i < contextSpectrum.length; i++) {
                posGradient[i] -= contextSpectrum[i];
            }

            // Negative gradient: push away from context
            const negGradient = new Float32Array(CONFIG.frequencyDim * 2);
            for (let i = 0; i < contextSpectrum.length; i++) {
                negGradient[i] += contextSpectrum[i];
            }

            updateSpectrum(negWord.spectrum, negGradient, CONFIG.learningRate, CONFIG.sparsityPenalty);
        }
    }

    // Update positive word
    if (hasLoss) {
        updateSpectrum(centerWord.spectrum, posGradient, CONFIG.learningRate, CONFIG.sparsityPenalty);
    }

    // Two-phase pruning
    state.updateStep = (state.updateStep || 0) + 1;
    const progress = state.totalDocuments / (state.estimatedTotalDocs || 1000);

    if (progress < CONFIG.explorationPhase) {
        // PHASE 1: Aggressive pruning every step
        pruneSpectrum(centerWord.spectrum, CONFIG.earlyPruneThreshold);
        for (const negWord of negativeWords) {
            pruneSpectrum(negWord.spectrum, CONFIG.earlyPruneThreshold);
        }
    } else {
        // PHASE 2: Gentle pruning every N steps
        if (state.updateStep % CONFIG.latePruneInterval === 0) {
            pruneSpectrum(centerWord.spectrum, CONFIG.latePruneThreshold);
            for (const negWord of negativeWords) {
                pruneSpectrum(negWord.spectrum, CONFIG.latePruneThreshold);
            }
        }
    }
}

// ============================================
// STATE MANAGEMENT
// ============================================

function loadCurrentState() {
    if (!fs.existsSync(CONFIG.stateFile)) {
        return {
            currentFileIndex: 0,
            currentDocument: 0,
            totalDocuments: 0,
            estimatedTotalDocs: 10000,
            updateStep: 0,
            parquetFiles: [],
            snapshots: [],              // Track all snapshots
            lastSnapshotDocs: 0         // Docs count at last snapshot
        };
    }

    try {
        const data = fs.readFileSync(CONFIG.stateFile, 'utf-8');
        const state = JSON.parse(data);

        // Ensure snapshots array exists (for backward compatibility)
        if (!state.snapshots) state.snapshots = [];
        if (!state.lastSnapshotDocs) state.lastSnapshotDocs = 0;

        return state;
    } catch (error) {
        console.error('Error loading state:', error.message);
        return {
            currentFileIndex: 0,
            currentDocument: 0,
            totalDocuments: 0,
            estimatedTotalDocs: 10000,
            updateStep: 0,
            parquetFiles: [],
            snapshots: [],
            lastSnapshotDocs: 0
        };
    }
}

function saveCurrentState(state) {
    const dir = path.dirname(CONFIG.stateFile);
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(CONFIG.stateFile, JSON.stringify(state, null, 2));
}

/**
 * Save model snapshot for analysis (polysemy/stability)
 */
function saveSnapshot(state) {
    // Create snapshot directory
    if (!fs.existsSync(CONFIG.snapshotDir)) {
        fs.mkdirSync(CONFIG.snapshotDir, { recursive: true });
    }

    const snapshotNumber = state.snapshots.length + 1;
    const filename = `snapshot_${String(snapshotNumber).padStart(4, '0')}_docs${state.totalDocuments}.json`;
    const filepath = path.join(CONFIG.snapshotDir, filename);

    // Build vocabulary array from vocabulary module
    // Use getAllWords() which properly iterates the vocabulary Map
    const vocabArray = vocabulary.getAllWords();

    const snapshot = {
        version: '1.0.0',
        timestamp: new Date().toISOString(),
        snapshotNumber: snapshotNumber,
        trainingProgress: {
            totalDocuments: state.totalDocuments,
            currentFile: state.currentFileIndex,
            phase: state.totalDocuments < state.estimatedTotalDocs * CONFIG.explorationPhase ? 'exploration' : 'refinement'
        },
        config: {
            frequencyDim: CONFIG.frequencyDim,
            maxFrequencies: CONFIG.maxFrequencies,
            windowSize: CONFIG.windowSize,
            learningRate: CONFIG.learningRate,
            sparsityPenalty: CONFIG.sparsityPenalty,
            negativeCount: CONFIG.negativeCount
        },
        vocabulary: vocabArray,
        stats: {
            vocabSize: vocabArray.length,
            avgSparsity: calculateAvgSparsity(vocabArray)
        }
    };

    fs.writeFileSync(filepath, JSON.stringify(snapshot, null, 2));

    // Update state with snapshot info
    state.snapshots.push({
        number: snapshotNumber,
        filename: filename,
        filepath: filepath,
        documents: state.totalDocuments,
        timestamp: snapshot.timestamp
    });

    state.lastSnapshotDocs = state.totalDocuments;

    console.log(`\n  📸 Snapshot ${snapshotNumber} saved: ${filename}`);
    console.log(`     Vocab: ${snapshot.stats.vocabSize}, Avg sparsity: ${snapshot.stats.avgSparsity.toFixed(2)}`);
}

/**
 * Get word string by ID (helper for snapshot)
 */
function getWordStringById(id) {
    return vocabulary.getWordString(id) || `word_${id}`;
}

/**
 * Calculate average sparsity across all words
 */
function calculateAvgSparsity(vocabArray) {
    let totalSparsity = 0;
    let count = 0;

    for (const wordObj of vocabArray) {
        if (wordObj.spectrum && wordObj.spectrum.frequencies) {
            totalSparsity += wordObj.spectrum.frequencies.length;
            count++;
        }
    }

    return count > 0 ? totalSparsity / count : 0;
}

// ============================================
// DATA LOADING
// ============================================

function loadParquetFile(filepath) {
    console.log(`Loading parquet: ${filepath}`);

    try {
        const pythonCmd = `python scripts/read_parquet.py "${filepath}"`;
        const output = execSync(pythonCmd, {
            encoding: 'utf8',
            maxBuffer: 500 * 1024 * 1024
        });

        const documents = JSON.parse(output);
        console.log(`  Loaded ${documents.length} documents`);
        return documents;
    } catch (error) {
        console.error(`Error loading parquet file: ${error.message}`);
        return [];
    }
}

function getParquetFiles() {
    if (!fs.existsSync(CONFIG.parquetDir)) {
        console.error(`Parquet directory not found: ${CONFIG.parquetDir}`);
        return [];
    }

    return fs.readdirSync(CONFIG.parquetDir)
        .filter(f => f.endsWith('.parquet'))
        .sort()
        .map(f => path.join(CONFIG.parquetDir, f));
}

// ============================================
// MAIN TRAINING LOOP
// ============================================

function train() {
    console.log('\n' + '='.repeat(70));
    console.log('CSS (COMPRESSIVE SEMANTIC SPECTROSCOPY) TRAINING');
    console.log('='.repeat(70) + '\n');

    console.log('Configuration:');
    console.log(`  Frequency dimension: ${CONFIG.frequencyDim}`);
    console.log(`  Max frequencies/word: ${CONFIG.maxFrequencies}`);
    console.log(`  Window size: ${CONFIG.windowSize}`);
    console.log(`  Negative samples: ${CONFIG.negativeCount}`);
    console.log(`  Initial amplitudes: [${CONFIG.initAmpMin}, ${CONFIG.initAmpMax}]`);
    console.log(`  Two-phase pruning: ${CONFIG.explorationPhase * 100}% exploration\n`);

    vocabulary.init();

    let state = loadCurrentState();
    globalTrainingState = state;  // Make state accessible to shutdown handler

    const parquetFiles = getParquetFiles();

    if (parquetFiles.length === 0) {
        console.error('No parquet files found!');
        process.exit(1);
    }

    if (state.parquetFiles.length === 0) {
        state.parquetFiles = parquetFiles;
    }

    console.log(`Found ${parquetFiles.length} parquet file(s)\n`);

    const startTime = Date.now();

    for (let fileIndex = state.currentFileIndex; fileIndex < parquetFiles.length; fileIndex++) {
        // Update global position
        currentFileIndex = fileIndex;

        // Check if shutdown was requested between files
        if (isShuttingDown) {
            return;
        }

        const filepath = parquetFiles[fileIndex];
        console.log(`\nProcessing file ${fileIndex + 1}/${parquetFiles.length}: ${path.basename(filepath)}`);

        const documents = loadParquetFile(filepath);
        if (documents.length === 0) continue;

        const startDoc = (fileIndex === state.currentFileIndex) ? state.currentDocument : 0;

        for (let docIndex = startDoc; docIndex < documents.length; docIndex++) {
            // Update global position
            currentDocIndex = docIndex;

            // Check if shutdown was requested
            if (isShuttingDown) {
                return;
            }

            const content = documents[docIndex];
            processDocument(content, state);

            state.totalDocuments++;

            if (state.totalDocuments % CONFIG.logEvery === 0) {
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                const docsPerSec = (state.totalDocuments / (Date.now() - startTime) * 1000).toFixed(1);
                const progress = (state.totalDocuments / state.estimatedTotalDocs * 100).toFixed(1);
                const phase = progress < CONFIG.explorationPhase * 100 ? 'EXPLORATION' : 'REFINEMENT';

                console.log(`  [${phase}] Docs: ${state.totalDocuments}, Vocab: ${vocabulary.getVocabSize()}, ${docsPerSec} docs/sec`);
            }

            if (state.totalDocuments % CONFIG.saveEvery === 0) {
                state.currentFileIndex = fileIndex;
                state.currentDocument = docIndex + 1;
                globalTrainingState = state;  // Update global state before saving
                saveCurrentState(state);
                vocabulary.saveVocabulary();
                console.log(`  Checkpoint saved (Ctrl+C safe)`);
            }

            // Save snapshot for analysis (polysemy/stability)
            if (CONFIG.snapshotEvery > 0 &&
                state.totalDocuments % CONFIG.snapshotEvery === 0 &&
                state.totalDocuments !== state.lastSnapshotDocs) {

                saveSnapshot(state);
                globalTrainingState = state;  // Update global state
                saveCurrentState(state);  // Update state with snapshot info
            }
        }

        // Exit if shutdown was requested
        if (isShuttingDown) {
            return;
        }

        state.currentFileIndex = fileIndex + 1;
        state.currentDocument = 0;
        globalTrainingState = state;  // Update global state
        saveCurrentState(state);
    }

    console.log('\n' + '='.repeat(70));
    console.log('Training complete!');
    console.log('='.repeat(70));

    // Save final snapshot if not already saved
    if (state.totalDocuments !== state.lastSnapshotDocs) {
        console.log('\nSaving final snapshot...');
        saveSnapshot(state);
    }

    // Final save of vocabulary and state
    console.log('Saving final vocabulary and state...');
    vocabulary.saveVocabulary();
    saveCurrentState(state);

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`\nTotal documents: ${state.totalDocuments}`);
    console.log(`Final vocabulary: ${vocabulary.getVocabSize()}`);
    console.log(`Time: ${elapsed}s`);

    // Show snapshot summary
    if (state.snapshots.length > 0) {
        console.log(`\nSnapshots saved: ${state.snapshots.length}`);
        state.snapshots.forEach((snap, idx) => {
            console.log(`  ${idx + 1}. ${snap.filename} (${snap.documents} docs)`);
        });

        console.log(`\nAnalyze training progress:`);
        console.log(`  # Compare first and last snapshot`);
        if (state.snapshots.length >= 2) {
            const first = state.snapshots[0];
            const last = state.snapshots[state.snapshots.length - 1];
            console.log(`  node src/analyze_stability.js "${first.filepath}" "${last.filepath}"`);
        }

        console.log(`\n  # Time series analysis`);
        console.log(`  node src/analyze_stability.js "${CONFIG.snapshotDir}/snapshot_*.json"`);
    }

    console.log('\n✅ Training completed successfully!\n');

    // Exit cleanly
    process.exit(0);
}

// ============================================
// EXPORT & ERROR HANDLING
// ============================================

// Track training state globally for shutdown handler
let globalTrainingState = null;
let isShuttingDown = false;
let currentFileIndex = 0;
let currentDocIndex = 0;

// Graceful shutdown handler
function gracefulShutdown(signal) {
    if (isShuttingDown) {
        console.log('\n\n⚠️  Force quit detected. Exiting immediately without saving!');
        console.log('Your progress may be lost. Use Ctrl+C once for graceful shutdown.\n');
        process.exit(1);
    }

    isShuttingDown = true;

    console.log(`\n\n⚠️  Training interrupted by ${signal}`);
    console.log('Waiting for current document to finish...');
    console.log('Saving current progress...\n');

    try {
        if (globalTrainingState) {
            // Update state with current position
            globalTrainingState.currentFileIndex = currentFileIndex;
            globalTrainingState.currentDocument = currentDocIndex;

            console.log('  Saving vocabulary...');
            vocabulary.saveVocabulary();
            console.log('  ✓ Vocabulary saved');

            console.log('  Saving training state...');
            saveCurrentState(globalTrainingState);
            console.log('  ✓ Training state saved');

            console.log('\n✅ Progress saved successfully!');
            console.log(`  Total documents processed: ${globalTrainingState.totalDocuments}`);
            console.log(`  Vocabulary size: ${vocabulary.getVocabSize()}`);
            console.log(`  Last position: File ${currentFileIndex + 1}, Document ${currentDocIndex}`);
            console.log('\nYou can resume training by running: npm run train\n');
        } else {
            console.log('⚠️  No training state found. Training may not have started yet.\n');
        }
    } catch (err) {
        console.error('✗ Failed to save progress:', err.message);
        console.error(err.stack);
        console.log('\nSome data may be lost. Check the error above.\n');
    }

    process.exit(0);
}

// Graceful shutdown on interruption (Ctrl+C)
process.on('SIGINT', () => gracefulShutdown('SIGINT (Ctrl+C)'));

// Handle SIGTERM (sent by some process managers)
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));

// Windows-specific: Better Ctrl+C handling
if (process.platform === 'win32') {
    // Use readline for better Windows Ctrl+C support
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    rl.on('SIGINT', () => {
        process.emit('SIGINT');
    });

    // Keep stdin open but don't show prompts
    process.stdin.on('data', () => {});
}

// Handle uncaught errors
process.on('uncaughtException', (err) => {
    console.error('\n\n❌ Fatal error during training:');
    console.error(err);
    console.log('\nAttempting to save current progress...');

    try {
        vocabulary.saveVocabulary();
        console.log('✓ Vocabulary saved');
    } catch (saveErr) {
        console.error('✗ Failed to save vocabulary:', saveErr.message);
    }

    console.log('\nTraining terminated.\n');
    process.exit(1);
});

// Run training
try {
    train();
} catch (err) {
    console.error('\n\n❌ Training failed:');
    console.error(err);

    console.log('\nAttempting to save current progress...');
    try {
        vocabulary.saveVocabulary();
        console.log('✓ Vocabulary saved');
    } catch (saveErr) {
        console.error('✗ Failed to save vocabulary:', saveErr.message);
    }

    console.log('\n');
    process.exit(1);
}
