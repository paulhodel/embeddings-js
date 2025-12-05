/**
 * Training Script - Model 2 (Skip-gram)
 * Standard skip-gram with negative sampling
 * Given a center word, predict context words
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

const CONFIG = {
    // Data
    parquetDir: './data/parquet',
    dictionaryFile: './data/dictionary.ndjson',
    checkpointDir: './data/checkpoints2',
    maxParquetFiles: 20,  // Limit number of files to process

    // Architecture
    embeddingDim: 64,
    contextWindow: 3,

    // Training
    learningRate: 0.025,
    batchSize: 20,
    epochs: 5,
    negativeSamples: 5,

    // Checkpointing
    checkpointEvery: 1000000,
};

/**
 * Sigmoid function
 */
function sigmoid(x) {
    return 1.0 / (1.0 + Math.exp(-x));
}

async function main() {
    console.log('\n' + '='.repeat(60));
    console.log('TRAINING - MODEL 2 (Skip-gram)');
    console.log('='.repeat(60));
    console.log('\nConfiguration:');
    console.log(`  Embedding dim:    ${CONFIG.embeddingDim}`);
    console.log(`  Context window:   ${CONFIG.contextWindow}`);
    console.log(`  Learning rate:    ${CONFIG.learningRate}`);
    console.log(`  Negative samples: ${CONFIG.negativeSamples}`);
    console.log(`  Epochs:           ${CONFIG.epochs}`);

    const startTime = Date.now();

    if (!fs.existsSync(CONFIG.checkpointDir)) {
        fs.mkdirSync(CONFIG.checkpointDir, { recursive: true });
    }

    // ============================================
    // LOAD DICTIONARY
    // ============================================
    console.log(`\nLoading dictionary from: ${CONFIG.dictionaryFile}`);

    const lines = fs.readFileSync(CONFIG.dictionaryFile, 'utf8').split('\n');
    const inputEmbeddings = [];  // Center word embeddings
    const outputEmbeddings = [];  // Context word embeddings
    const wordToId = new Map();
    const idToWord = [];
    const frequencies = [];

    let lineCount = 0;
    const embDim = CONFIG.embeddingDim;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line.length === 0) continue;

        const parts = line.split('\t');
        const word = parts[0];
        const id = parseInt(parts[1]);
        const frequency = parseInt(parts[2]);
        const vectorStr = parts[3];

        const vectorParts = vectorStr.split(' ');
        const vector = new Array(embDim);
        for (let j = 0; j < embDim; j++) {
            vector[j] = parseFloat(vectorParts[j]);
        }

        inputEmbeddings[id] = vector;

        // Initialize output embeddings randomly
        const outVector = new Array(embDim);
        for (let j = 0; j < embDim; j++) {
            outVector[j] = (Math.random() * 0.2) - 0.1;
        }
        outputEmbeddings[id] = outVector;

        wordToId.set(word, id);
        idToWord[id] = word;
        frequencies[id] = frequency;
        lineCount++;
    }

    const vocabSize = inputEmbeddings.length;
    console.log(`Loaded ${lineCount} words`);

    // Build negative sampling distribution (frequency^0.75)
    const negSamplingProbs = new Array(vocabSize);
    let totalProb = 0;
    for (let i = 0; i < vocabSize; i++) {
        negSamplingProbs[i] = Math.pow(frequencies[i], 0.75);
        totalProb += negSamplingProbs[i];
    }
    for (let i = 0; i < vocabSize; i++) {
        negSamplingProbs[i] /= totalProb;
    }

    // Cumulative distribution for sampling
    const cumulativeProbs = new Array(vocabSize);
    cumulativeProbs[0] = negSamplingProbs[0];
    for (let i = 1; i < vocabSize; i++) {
        cumulativeProbs[i] = cumulativeProbs[i - 1] + negSamplingProbs[i];
    }

    console.log('Negative sampling distribution built');

    // ============================================
    // CHECK FOR EXISTING CHECKPOINT AND PROGRESS
    // ============================================
    let startEpoch = 0;
    let startFileIdx = 0;
    let startDocIdx = 0;
    let batchCount = 0;
    let lastCheckpointTime = Date.now();

    const progressFile = path.join(CONFIG.checkpointDir, 'training_progress.json');

    // Load progress tracker first (lightweight)
    if (fs.existsSync(progressFile)) {
        const progress = JSON.parse(fs.readFileSync(progressFile, 'utf8'));
        startEpoch = progress.epoch;
        startFileIdx = progress.fileIdx;
        startDocIdx = progress.docIdx;
        batchCount = progress.batch;
        console.log(`\nFound progress: epoch ${startEpoch + 1}, file ${startFileIdx}, doc ${startDocIdx}, batch ${batchCount}`);
    }

    // Load checkpoint for embeddings (heavy)
    const checkpointFiles = fs.existsSync(CONFIG.checkpointDir)
        ? fs.readdirSync(CONFIG.checkpointDir).filter(f => f.startsWith('checkpoint_epoch_'))
        : [];

    if (checkpointFiles.length > 0) {
        // Sort numerically and load latest
        checkpointFiles.sort((a, b) => {
            const batchA = parseInt(a.match(/batch_(\d+)/)[1]);
            const batchB = parseInt(b.match(/batch_(\d+)/)[1]);
            return batchB - batchA;  // Descending
        });

        const latestCheckpoint = checkpointFiles[0];
        const checkpointPath = path.join(CONFIG.checkpointDir, latestCheckpoint);

        console.log(`Loading checkpoint: ${latestCheckpoint}`);

        const checkpoint = JSON.parse(fs.readFileSync(checkpointPath, 'utf8'));

        // Restore embeddings
        for (let i = 0; i < vocabSize; i++) {
            inputEmbeddings[i] = checkpoint.embeddings[i];
        }

        if (checkpoint.outputEmbeddings) {
            for (let i = 0; i < vocabSize; i++) {
                outputEmbeddings[i] = checkpoint.outputEmbeddings[i];
            }
        }

        console.log(`Resuming from epoch ${startEpoch + 1}, file ${startFileIdx + 1}, doc ${startDocIdx + 1}, batch ${batchCount}`);
    }

    // ============================================
    // FIND PARQUET FILES
    // ============================================
    console.log(`\nScanning parquet files from: ${CONFIG.parquetDir}`);

    const allParquetFiles = fs.readdirSync(CONFIG.parquetDir)
        .filter(f => f.endsWith('.parquet'))
        .sort();

    const maxFiles = Math.min(CONFIG.maxParquetFiles, allParquetFiles.length);
    const parquetFiles = allParquetFiles.slice(0, maxFiles);

    console.log(`Found ${allParquetFiles.length} parquet files, will process ${parquetFiles.length}`);

    // ============================================
    // TRAINING LOOP - Process one file at a time
    // ============================================
    console.log('\n' + '='.repeat(60));
    console.log('STARTING TRAINING');
    console.log('='.repeat(60));

    const contextSize = CONFIG.contextWindow;
    const lr = CONFIG.learningRate;
    const batchSize = CONFIG.batchSize;
    const negSamples = CONFIG.negativeSamples;

    // Gradient accumulators
    const gradInput = new Map();
    const gradOutput = new Map();
    let batchWindowCount = 0;
    let pairsProcessed = 0;  // Track total pairs for throughput comparison

    const embDimCached = embDim;
    const vocabSizeCached = vocabSize;
    const negSamplesCached = negSamples;
    const batchSizeCached = batchSize;
    const parquetFilesLen = parquetFiles.length;

    for (let epoch = startEpoch; epoch < CONFIG.epochs; epoch++) {
        console.log(`\nEpoch ${epoch + 1}/${CONFIG.epochs}`);
        let epochLoss = 0;
        let windowsProcessed = 0;

        // Process each parquet file
        for (let fileIdx = startFileIdx; fileIdx < parquetFilesLen; fileIdx++) {
            const filepath = path.join(CONFIG.parquetDir, parquetFiles[fileIdx]);
            console.log(`\n[File ${fileIdx + 1}/${parquetFilesLen}] Loading ${parquetFiles[fileIdx]}...`);

            // Load one file at a time
            let fileTexts = [];
            try {
                const pythonCmd = `python scripts/read_parquet.py "${filepath}"`;
                const output = execSync(pythonCmd, {
                    encoding: 'utf8',
                    maxBuffer: 500 * 1024 * 1024
                });
                fileTexts = JSON.parse(output);
                console.log(`  Loaded ${fileTexts.length} documents`);
            } catch (error) {
                console.error(`  Error reading ${filepath}:`, error.message);
                continue;
            }

            // Tokenize documents from this file inline
            console.log('  Tokenizing...');
            const tokenizedDocs = [];
            const numDocs = fileTexts.length;

            for (let docIdx = 0; docIdx < numDocs; docIdx++) {
                const text = fileTexts[docIdx];
                const textLen = text.length;
                const tokens = [];

                let currentWord = '';
                let hasLetter = false;

                for (let charIdx = 0; charIdx < textLen; charIdx++) {
                    const code = text.charCodeAt(charIdx);

                    if ((code >= 65 && code <= 90) || (code >= 97 && code <= 122)) {
                        if (code >= 65 && code <= 90) {
                            currentWord += String.fromCharCode(code + 32);
                        } else {
                            currentWord += text[charIdx];
                        }
                        hasLetter = true;
                    } else if (code >= 48 && code <= 57) {
                        currentWord += text[charIdx];
                    } else {
                        if (hasLetter && currentWord.length >= 2) {
                            const wordId = wordToId.get(currentWord);
                            if (wordId !== undefined) {
                                tokens.push(wordId);
                            }
                        }
                        currentWord = '';
                        hasLetter = false;
                    }
                }

                if (hasLetter && currentWord.length >= 2) {
                    const wordId = wordToId.get(currentWord);
                    if (wordId !== undefined) {
                        tokens.push(wordId);
                    }
                }

                if (tokens.length > 0) {
                    tokenizedDocs.push(tokens);
                }
            }

            console.log(`  Tokenized ${tokenizedDocs.length} documents, starting training...`);
            fileTexts = null;  // Free memory

            // Train on this file's documents
            const numTokenizedDocs = tokenizedDocs.length;
            const docStartIdx = (fileIdx === startFileIdx) ? startDocIdx : 0;

            for (let docIdx = docStartIdx; docIdx < numTokenizedDocs; docIdx++) {
                const tokens = tokenizedDocs[docIdx];
            const numTokens = tokens.length;

            for (let center = contextSize; center < numTokens - contextSize; center++) {
                const centerId = tokens[center];
                const centerVec = inputEmbeddings[centerId];

                // Process all context words inline
                for (let offset = -contextSize; offset <= contextSize; offset++) {
                    if (offset === 0) continue;

                    const contextId = tokens[center + offset];
                    const contextVec = outputEmbeddings[contextId];

                    // POSITIVE SAMPLE - inline dot product and gradient
                    let dot = 0;
                    for (let d = 0; d < embDimCached; d++) {
                        dot += centerVec[d] * contextVec[d];
                    }

                    const pred = 1.0 / (1.0 + Math.exp(-dot));  // sigmoid inline
                    epochLoss += -Math.log(pred + 1e-8);
                    const grad = pred - 1.0;

                    // Get or create gradient arrays inline
                    let gIn = gradInput.get(centerId);
                    if (gIn === undefined) {
                        gIn = new Array(embDimCached).fill(0);
                        gradInput.set(centerId, gIn);
                    }
                    let gOut = gradOutput.get(contextId);
                    if (gOut === undefined) {
                        gOut = new Array(embDimCached).fill(0);
                        gradOutput.set(contextId, gOut);
                    }

                    // Accumulate positive gradients inline
                    for (let d = 0; d < embDimCached; d++) {
                        gIn[d] += grad * contextVec[d];
                        gOut[d] += grad * centerVec[d];
                    }

                    // NEGATIVE SAMPLES - inline
                    for (let n = 0; n < negSamplesCached; n++) {
                        // Binary search for negative sample
                        const r = Math.random();
                        let negId = 0;
                        for (let i = 0; i < vocabSizeCached; i++) {
                            if (r < cumulativeProbs[i]) {
                                negId = i;
                                break;
                            }
                        }

                        if (negId === contextId) continue;

                        const negVec = outputEmbeddings[negId];

                        // Negative dot product inline
                        let negDot = 0;
                        for (let d = 0; d < embDimCached; d++) {
                            negDot += centerVec[d] * negVec[d];
                        }

                        const negPred = 1.0 / (1.0 + Math.exp(-negDot));  // sigmoid inline
                        epochLoss += -Math.log(1.0 - negPred + 1e-8);
                        const negGrad = negPred;

                        // Get or create negative gradient array
                        let gNeg = gradOutput.get(negId);
                        if (gNeg === undefined) {
                            gNeg = new Array(embDimCached).fill(0);
                            gradOutput.set(negId, gNeg);
                        }

                        // Accumulate negative gradients inline
                        for (let d = 0; d < embDimCached; d++) {
                            gIn[d] += negGrad * negVec[d];
                            gNeg[d] += negGrad * centerVec[d];
                        }
                    }

                    windowsProcessed++;
                    batchWindowCount++;
                    batchCount++;
                    pairsProcessed++;  // Each context word = 1 training pair

                    // Apply batch update
                    if (batchWindowCount === batchSize) {
                        // Update input embeddings and normalize
                        for (const [id, grad] of gradInput.entries()) {
                            const vec = inputEmbeddings[id];
                            for (let d = 0; d < embDim; d++) {
                                vec[d] -= lr * (grad[d] / batchSize);
                            }

                            // L2 normalize
                            let norm = 0;
                            for (let d = 0; d < embDim; d++) {
                                norm += vec[d] * vec[d];
                            }
                            norm = Math.sqrt(norm) + 1e-8;
                            for (let d = 0; d < embDim; d++) {
                                vec[d] /= norm;
                            }
                        }

                        // Update output embeddings
                        for (const [id, grad] of gradOutput.entries()) {
                            const vec = outputEmbeddings[id];
                            for (let d = 0; d < embDim; d++) {
                                vec[d] -= lr * (grad[d] / batchSize);
                            }
                        }

                        gradInput.clear();
                        gradOutput.clear();
                        batchWindowCount = 0;
                    }

                    // Checkpoint
                    if (batchCount % CONFIG.checkpointEvery === 0) {
                        const avgLoss = epochLoss / windowsProcessed;
                        const currentTime = Date.now();
                        const timeSinceLastCheckpoint = (currentTime - lastCheckpointTime) / 1000;
                        const pairsPerSecond = Math.floor(CONFIG.checkpointEvery / timeSinceLastCheckpoint);

                        console.log('\n' + '='.repeat(60));
                        console.log(`CHECKPOINT at ${pairsProcessed} pairs`);
                        console.log('='.repeat(60));
                        console.log(`Epoch: ${epoch + 1}/${CONFIG.epochs}`);
                        console.log(`File: ${fileIdx + 1}/${parquetFilesLen}, Doc: ${docIdx + 1}/${numTokenizedDocs}`);
                        console.log(`Pairs: ${pairsProcessed}`);
                        console.log(`Avg Loss: ${avgLoss.toFixed(6)}`);
                        console.log(`Throughput: ${pairsPerSecond} pairs/s`);
                        console.log(`Time: ${timeSinceLastCheckpoint.toFixed(1)}s`);

                        // Save lightweight progress file
                        const progress = {
                            epoch: epoch,
                            fileIdx: fileIdx,
                            docIdx: docIdx,
                            batch: batchCount,
                            timestamp: Date.now()
                        };
                        fs.writeFileSync(progressFile, JSON.stringify(progress));

                        // Save full checkpoint (heavy)
                        const checkpoint = {
                            epoch: epoch,
                            batch: batchCount,
                            currentFileIdx: fileIdx,
                            embeddings: inputEmbeddings,
                            outputEmbeddings: outputEmbeddings,
                            wordToId: Array.from(wordToId.entries()),
                            idToWord: idToWord,
                            vocabSize: vocabSize,
                            avgLoss: avgLoss,
                            timestamp: Date.now()
                        };

                        const checkpointPath = path.join(
                            CONFIG.checkpointDir,
                            `checkpoint_epoch_${epoch}_batch_${batchCount}.json`
                        );

                        fs.writeFileSync(checkpointPath, JSON.stringify(checkpoint));
                        console.log(`Checkpoint saved to: ${checkpointPath}`);
                        console.log('='.repeat(60) + '\n');

                        lastCheckpointTime = currentTime;
                    }
                }
            }

            if ((docIdx + 1) % 5000 === 0) {
                console.log(`  Processed ${docIdx + 1}/${numTokenizedDocs} documents`);
            }
        }

        console.log(`  File ${fileIdx + 1}/${parquetFilesLen} complete`);
    }

    // Reset startFileIdx after completing all files in an epoch
    startFileIdx = 0;

        // Apply remaining gradients at end of epoch
        if (batchWindowCount > 0) {
            for (const [id, grad] of gradInput.entries()) {
                const vec = inputEmbeddings[id];
                for (let d = 0; d < embDim; d++) {
                    vec[d] -= lr * (grad[d] / batchWindowCount);
                }

                let norm = 0;
                for (let d = 0; d < embDim; d++) {
                    norm += vec[d] * vec[d];
                }
                norm = Math.sqrt(norm) + 1e-8;
                for (let d = 0; d < embDim; d++) {
                    vec[d] /= norm;
                }
            }

            for (const [id, grad] of gradOutput.entries()) {
                const vec = outputEmbeddings[id];
                for (let d = 0; d < embDim; d++) {
                    vec[d] -= lr * (grad[d] / batchWindowCount);
                }
            }

            gradInput.clear();
            gradOutput.clear();
            batchWindowCount = 0;
        }

        const avgEpochLoss = epochLoss / windowsProcessed;
        console.log(`\nEpoch ${epoch + 1} complete | Avg Loss: ${avgEpochLoss.toFixed(6)}`);
    }

    // ============================================
    // SAVE FINAL MODEL
    // ============================================
    console.log('\nSaving final model...');

    const finalPath = './data/model2_final.ndjson';
    const writeStream = fs.createWriteStream(finalPath);

    const precision = 4;
    for (let i = 0; i < vocabSize; i++) {
        const word = idToWord[i];
        const vec = inputEmbeddings[i];

        let vectorStr = '';
        for (let j = 0; j < embDim; j++) {
            if (j > 0) vectorStr += ' ';
            vectorStr += vec[j].toFixed(precision);
        }

        const line = `${word}\t${i}\t${vectorStr}\n`;
        writeStream.write(line);
    }

    writeStream.end();

    await new Promise((resolve, reject) => {
        writeStream.on('finish', resolve);
        writeStream.on('error', reject);
    });

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`\nTraining complete! Time elapsed: ${elapsed}s`);
    console.log(`Final model saved to: ${finalPath}`);
}

main().catch(error => {
    console.error('\n❌ Training failed:', error.message);
    console.error(error.stack);
    process.exit(1);
});
