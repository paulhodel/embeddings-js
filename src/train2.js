/**
 * GPU-Optimized Training Script - Model 2 (Skip-gram)
 * Using gpu.js for WebGL GPU acceleration with MASSIVE batching
 */

import { GPU } from 'gpu.js';
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

const CONFIG = {
    // Data
    parquetDir: './data/parquet',
    dictionaryFile: './data/dictionary.ndjson',
    checkpointDir: './data/checkpoints2',
    maxParquetFiles: 20,

    // Architecture
    embeddingDim: 64,
    contextWindow: 3,

    // Training - AGGRESSIVE GPU settings
    learningRate: 0.025,
    gpuBatchSize: 50000,      // Process 50K windows at once on GPU
    windowsPerFile: 5000000,  // Pre-extract 5M windows from multiple files
    updateEvery: 10,          // Apply gradients every 10 GPU batches (500K windows)
    epochs: 5,
    negativeSamples: 5,

    // Checkpointing
    checkpointEvery: 10000000,
};

/**
 * Tokenize text
 */
function tokenizeText(text, wordToId) {
    const tokens = [];
    let currentWord = '';
    let hasLetter = false;

    const textLen = text.length;
    for (let i = 0; i < textLen; i++) {
        const code = text.charCodeAt(i);

        if ((code >= 65 && code <= 90) || (code >= 97 && code <= 122)) {
            currentWord += (code >= 65 && code <= 90) ? String.fromCharCode(code + 32) : text[i];
            hasLetter = true;
        } else if (code >= 48 && code <= 57) {
            currentWord += text[i];
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

    return tokens;
}

/**
 * Extract training windows from a document
 */
function extractWindows(tokens, contextSize) {
    const windows = [];
    const numTokens = tokens.length;

    for (let center = contextSize; center < numTokens - contextSize; center++) {
        const centerId = tokens[center];

        for (let offset = -contextSize; offset <= contextSize; offset++) {
            if (offset === 0) continue;

            const contextId = tokens[center + offset];
            windows.push([centerId, contextId]);
        }
    }

    return windows;
}

/**
 * Sample negative word IDs using cumulative distribution
 */
function sampleNegatives(count, vocabSize, cumulativeProbs, excludeId) {
    const negatives = [];
    let attempts = 0;
    const maxAttempts = count * 10;

    while (negatives.length < count && attempts < maxAttempts) {
        attempts++;
        const r = Math.random();

        // Binary search for faster sampling
        let left = 0;
        let right = vocabSize - 1;
        let negId = 0;

        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
            if (r < cumulativeProbs[mid]) {
                negId = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        if (negId !== excludeId) {
            negatives.push(negId);
        }
    }

    // Fill with random if needed
    while (negatives.length < count) {
        const negId = Math.floor(Math.random() * vocabSize);
        if (negId !== excludeId) {
            negatives.push(negId);
        }
    }

    return negatives;
}

async function main() {
    console.log('\n' + '='.repeat(60));
    console.log('TRAINING - MODEL 2 (gpu.js MASSIVE BATCH GPU)');
    console.log('='.repeat(60));
    console.log('\nConfiguration:');
    console.log(`  Embedding dim:    ${CONFIG.embeddingDim}`);
    console.log(`  Context window:   ${CONFIG.contextWindow}`);
    console.log(`  Learning rate:    ${CONFIG.learningRate}`);
    console.log(`  GPU batch size:   ${CONFIG.gpuBatchSize.toLocaleString()} windows`);
    console.log(`  Windows per file: ${CONFIG.windowsPerFile.toLocaleString()}`);
    console.log(`  Update every:     ${CONFIG.updateEvery} batches (${(CONFIG.gpuBatchSize * CONFIG.updateEvery).toLocaleString()} windows)`);
    console.log(`  Negative samples: ${CONFIG.negativeSamples}`);
    console.log(`  Epochs:           ${CONFIG.epochs}`);

    // Initialize GPU
    const gpu = new GPU({ mode: 'gpu' });
    console.log(`\nGPU initialized: ${gpu.mode}`);

    const startTime = Date.now();

    if (!fs.existsSync(CONFIG.checkpointDir)) {
        fs.mkdirSync(CONFIG.checkpointDir, { recursive: true });
    }

    // ============================================
    // LOAD DICTIONARY
    // ============================================
    console.log(`\nLoading dictionary from: ${CONFIG.dictionaryFile}`);

    const lines = fs.readFileSync(CONFIG.dictionaryFile, 'utf8').split('\n');
    const inputEmbeddingsList = [];
    const outputEmbeddingsList = [];
    const wordToId = new Map();
    const idToWord = [];
    const frequencies = [];

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
        const vector = new Float32Array(embDim);
        for (let j = 0; j < embDim; j++) {
            vector[j] = parseFloat(vectorParts[j]);
        }

        inputEmbeddingsList.push(vector);

        // Initialize output embeddings randomly
        const outVector = new Float32Array(embDim);
        for (let j = 0; j < embDim; j++) {
            outVector[j] = (Math.random() * 0.2) - 0.1;
        }
        outputEmbeddingsList.push(outVector);

        wordToId.set(word, id);
        idToWord[id] = word;
        frequencies[id] = frequency;
    }

    const vocabSize = inputEmbeddingsList.length;
    console.log(`Loaded ${vocabSize} words`);

    // Convert to flat arrays for GPU
    let inputEmbeddings = inputEmbeddingsList;
    let outputEmbeddings = outputEmbeddingsList;

    // Build negative sampling distribution
    const negSamplingProbs = new Float32Array(vocabSize);
    let totalProb = 0;
    for (let i = 0; i < vocabSize; i++) {
        negSamplingProbs[i] = Math.pow(frequencies[i], 0.75);
        totalProb += negSamplingProbs[i];
    }
    for (let i = 0; i < vocabSize; i++) {
        negSamplingProbs[i] /= totalProb;
    }

    // Cumulative distribution
    const cumulativeProbs = new Float32Array(vocabSize);
    cumulativeProbs[0] = negSamplingProbs[0];
    for (let i = 1; i < vocabSize; i++) {
        cumulativeProbs[i] = cumulativeProbs[i - 1] + negSamplingProbs[i];
    }

    console.log('Negative sampling distribution built');

    // ============================================
    // CREATE GPU KERNELS
    // ============================================
    console.log('\nCompiling GPU kernels...');

    // Kernel: Compute dot products for a batch
    const computeDotProducts = gpu.createKernel(function(inputEmbs, outputEmbs, centerIds, contextIds, embDim, vocabSize) {
        const i = this.thread.x;
        const centerId = centerIds[i];
        const contextId = contextIds[i];

        let dot = 0;
        for (let d = 0; d < embDim; d++) {
            dot += inputEmbs[centerId * embDim + d] * outputEmbs[contextId * embDim + d];
        }

        return dot;
    })
        .setOutput([CONFIG.gpuBatchSize])
        .setPipeline(true);

    // Kernel: Compute gradients
    const computeGradients = gpu.createKernel(function(
        inputEmbs, outputEmbs, centerIds, contextIds, dots,
        negIds, embDim, vocabSize, negSamples, isPositive
    ) {
        const i = this.thread.x;
        const d = this.thread.y;

        const centerId = centerIds[i];
        const contextId = isPositive > 0 ? contextIds[i] : negIds[i * negSamples + Math.floor(d / embDim)];

        const dot = dots[i];
        const pred = 1.0 / (1.0 + Math.exp(-dot));
        const grad = isPositive > 0 ? (pred - 1.0) : pred;

        return grad * (this.thread.y < embDim ?
            outputEmbs[contextId * embDim + d] :
            inputEmbs[centerId * embDim + d]);
    })
        .setOutput([CONFIG.gpuBatchSize, embDim]);

    console.log('GPU kernels compiled');

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
    // TRAINING LOOP
    // ============================================
    console.log('\n' + '='.repeat(60));
    console.log('STARTING TRAINING');
    console.log('='.repeat(60));

    const contextSize = CONFIG.contextWindow;
    const lr = CONFIG.learningRate;
    const gpuBatchSize = CONFIG.gpuBatchSize;
    const updateEvery = CONFIG.updateEvery;
    const negSamples = CONFIG.negativeSamples;
    const windowsPerFile = CONFIG.windowsPerFile;

    let batchCount = 0;
    let windowsProcessed = 0;
    let epochLoss = 0;
    let lastCheckpointTime = Date.now();
    let lastProgressTime = Date.now();

    // Gradient accumulators
    const gradInput = new Array(vocabSize);
    const gradOutput = new Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
        gradInput[i] = new Float32Array(embDim);
        gradOutput[i] = new Float32Array(embDim);
    }
    let gradCount = 0;

    for (let epoch = 0; epoch < CONFIG.epochs; epoch++) {
        console.log(`\nEpoch ${epoch + 1}/${CONFIG.epochs}`);

        let fileIdx = 0;

        while (fileIdx < parquetFiles.length) {
            // Load multiple files and extract windows until we hit target
            console.log(`\n[Extracting ${windowsPerFile.toLocaleString()} windows from files ${fileIdx + 1}+...]`);

            const allWindows = [];
            const extractStart = Date.now();

            while (allWindows.length < windowsPerFile && fileIdx < parquetFiles.length) {
                const filename = parquetFiles[fileIdx];
                const filepath = path.join(CONFIG.parquetDir, filename);

                console.log(`  Loading ${filename}...`);

                // Load one file
                let fileTexts = [];
                try {
                    const pythonCmd = `python scripts/read_parquet.py "${filepath}"`;
                    const output = execSync(pythonCmd, {
                        encoding: 'utf8',
                        maxBuffer: 500 * 1024 * 1024
                    });
                    fileTexts = JSON.parse(output);
                    console.log(`    Loaded ${fileTexts.length} documents`);
                } catch (error) {
                    console.error(`    Error reading ${filepath}:`, error.message);
                    fileIdx++;
                    continue;
                }

                // Tokenize and extract windows
                for (let docIdx = 0; docIdx < fileTexts.length; docIdx++) {
                    const tokens = tokenizeText(fileTexts[docIdx], wordToId);
                    if (tokens.length > contextSize * 2) {
                        const windows = extractWindows(tokens, contextSize);
                        // Push individually to avoid stack overflow
                        for (let w = 0; w < windows.length; w++) {
                            allWindows.push(windows[w]);
                            if (allWindows.length >= windowsPerFile) break;
                        }

                        if (allWindows.length >= windowsPerFile) break;
                    }
                }

                fileTexts = null;  // Free memory
                fileIdx++;

                console.log(`    Extracted ${allWindows.length.toLocaleString()} windows so far...`);
            }

            const extractTime = ((Date.now() - extractStart) / 1000).toFixed(1);
            console.log(`  Extraction complete: ${allWindows.length.toLocaleString()} windows in ${extractTime}s`);

            const numWindows = allWindows.length;
            if (numWindows === 0) continue;

            // Pre-sample ALL negative samples for this batch of windows
            console.log(`  Pre-sampling ${(numWindows * negSamples).toLocaleString()} negative samples...`);
            const allNegatives = new Int32Array(numWindows * negSamples);
            for (let i = 0; i < numWindows; i++) {
                const contextId = allWindows[i][1];
                const negs = sampleNegatives(negSamples, vocabSize, cumulativeProbs, contextId);
                for (let n = 0; n < negSamples; n++) {
                    allNegatives[i * negSamples + n] = negs[n];
                }

                if (i % 100000 === 0 && i > 0) {
                    console.log(`    Sampled ${i.toLocaleString()}/${numWindows.toLocaleString()} (${Math.floor(i/numWindows*100)}%)`);
                }
            }
            console.log(`  Negative sampling complete`);

            // Process windows in GPU batches
            console.log(`\n  Training on ${numWindows.toLocaleString()} windows with GPU...`);

            for (let batchStart = 0; batchStart < numWindows; batchStart += gpuBatchSize) {
                const batchEnd = Math.min(batchStart + gpuBatchSize, numWindows);
                const currentBatchSize = batchEnd - batchStart;

                // Prepare batch arrays
                const centerIds = new Int32Array(currentBatchSize);
                const contextIds = new Int32Array(currentBatchSize);
                const negativeIds = new Int32Array(currentBatchSize * negSamples);

                for (let i = 0; i < currentBatchSize; i++) {
                    centerIds[i] = allWindows[batchStart + i][0];
                    contextIds[i] = allWindows[batchStart + i][1];

                    for (let n = 0; n < negSamples; n++) {
                        negativeIds[i * negSamples + n] = allNegatives[(batchStart + i) * negSamples + n];
                    }
                }

                // Process positive samples (CPU for now - GPU kernels need more work)
                for (let i = 0; i < currentBatchSize; i++) {
                    const centerId = centerIds[i];
                    const contextId = contextIds[i];
                    const centerVec = inputEmbeddings[centerId];
                    const contextVec = outputEmbeddings[contextId];

                    // Positive sample
                    let dot = 0;
                    for (let d = 0; d < embDim; d++) {
                        dot += centerVec[d] * contextVec[d];
                    }

                    const pred = 1.0 / (1.0 + Math.exp(-dot));
                    epochLoss += -Math.log(pred + 1e-8);
                    const grad = pred - 1.0;

                    for (let d = 0; d < embDim; d++) {
                        gradInput[centerId][d] += grad * contextVec[d];
                        gradOutput[contextId][d] += grad * centerVec[d];
                    }

                    // Negative samples
                    for (let n = 0; n < negSamples; n++) {
                        const negId = negativeIds[i * negSamples + n];
                        const negVec = outputEmbeddings[negId];

                        let negDot = 0;
                        for (let d = 0; d < embDim; d++) {
                            negDot += centerVec[d] * negVec[d];
                        }

                        const negPred = 1.0 / (1.0 + Math.exp(-negDot));
                        epochLoss += -Math.log(1.0 - negPred + 1e-8);

                        for (let d = 0; d < embDim; d++) {
                            gradInput[centerId][d] += negPred * negVec[d];
                            gradOutput[negId][d] += negPred * centerVec[d];
                        }
                    }
                }

                windowsProcessed += currentBatchSize;
                batchCount += currentBatchSize;
                gradCount++;

                // Progress update every 5 seconds
                if (Date.now() - lastProgressTime > 5000) {
                    const progress = ((batchStart + currentBatchSize) / numWindows * 100).toFixed(1);
                    const currentTime = Date.now();
                    const timeSinceLast = (currentTime - lastCheckpointTime) / 1000;
                    const windowsSinceLast = windowsProcessed % CONFIG.checkpointEvery || windowsProcessed;
                    const currentSpeed = Math.floor(windowsSinceLast / timeSinceLast);
                    console.log(`    Progress: ${progress}% | Speed: ${currentSpeed.toLocaleString()} windows/s`);
                    lastProgressTime = currentTime;
                }

                // Apply gradients
                if (gradCount >= updateEvery) {
                    const gradScale = 1.0 / (gradCount * gpuBatchSize);

                    for (let i = 0; i < vocabSize; i++) {
                        for (let d = 0; d < embDim; d++) {
                            inputEmbeddings[i][d] -= lr * gradInput[i][d] * gradScale;
                            outputEmbeddings[i][d] -= lr * gradOutput[i][d] * gradScale;
                        }

                        // L2 normalize input
                        let norm = 0;
                        for (let d = 0; d < embDim; d++) {
                            norm += inputEmbeddings[i][d] * inputEmbeddings[i][d];
                        }
                        norm = Math.sqrt(norm) + 1e-8;
                        for (let d = 0; d < embDim; d++) {
                            inputEmbeddings[i][d] /= norm;
                        }

                        // Reset gradients
                        gradInput[i].fill(0);
                        gradOutput[i].fill(0);
                    }

                    gradCount = 0;
                }

                // Checkpoint
                if (batchCount % CONFIG.checkpointEvery === 0) {
                    const avgLoss = epochLoss / windowsProcessed;
                    const currentTime = Date.now();
                    const timeSinceLastCheckpoint = (currentTime - lastCheckpointTime) / 1000;
                    const windowsPerSecond = Math.floor(CONFIG.checkpointEvery / timeSinceLastCheckpoint);

                    console.log('\n' + '='.repeat(60));
                    console.log(`CHECKPOINT at batch ${batchCount.toLocaleString()}`);
                    console.log('='.repeat(60));
                    console.log(`Epoch: ${epoch + 1}/${CONFIG.epochs}`);
                    console.log(`Files processed: ${fileIdx}/${parquetFiles.length}`);
                    console.log(`Windows: ${windowsProcessed.toLocaleString()}`);
                    console.log(`Avg Loss: ${avgLoss.toFixed(6)}`);
                    console.log(`Time: ${timeSinceLastCheckpoint.toFixed(1)}s (${windowsPerSecond.toLocaleString()} windows/s)`);

                    // Save checkpoint
                    const checkpoint = {
                        epoch: epoch,
                        batch: batchCount,
                        embeddings: inputEmbeddings.map(v => Array.from(v)),
                        outputEmbeddings: outputEmbeddings.map(v => Array.from(v)),
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
                    console.log(`Saved: ${checkpointPath}`);
                    console.log('='.repeat(60) + '\n');

                    lastCheckpointTime = currentTime;
                }
            }

            console.log(`  Batch complete`);
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

    for (let i = 0; i < vocabSize; i++) {
        const word = idToWord[i];
        const vec = inputEmbeddings[i];

        let vectorStr = '';
        for (let j = 0; j < embDim; j++) {
            if (j > 0) vectorStr += ' ';
            vectorStr += vec[j].toFixed(4);
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
    const totalElapsed = ((Date.now() - startTime) / 60).toFixed(1);
    console.log(`\nTraining complete! Time elapsed: ${elapsed}s (${totalElapsed} minutes)`);
    console.log(`Final model saved to: ${finalPath}`);
    console.log(`Average speed: ${Math.floor(windowsProcessed / (Date.now() - startTime) * 1000).toLocaleString()} windows/s`);

    gpu.destroy();
}

main().catch(error => {
    console.error('\n❌ Training failed:', error.message);
    console.error(error.stack);
    process.exit(1);
});
