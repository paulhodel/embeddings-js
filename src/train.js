/**
 * Training Script - Model 1
 * 2-layer neural network with ReLU activation
 * Architecture: Input(1024) → Hidden1(1024) → Hidden2(1024) → Output(1024)
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

const CONFIG = {
    // Data
    parquetDir: './data/parquet',
    dictionaryFile: './data/dictionary.ndjson',
    checkpointDir: './data/checkpoints',

    // Architecture
    embeddingDim: 64,
    hiddenDim: 64,
    contextWindow: 3,

    // Training
    learningRate: 0.01,
    batchSize: 20,
    epochs: 5,

    // Checkpointing
    checkpointEvery: 1000000,
};

// ============================================
// HELPER FUNCTIONS
// ============================================

/**
 * He initialization for ReLU networks
 */
function initializeWeights(inputSize, outputSize) {
    const weights = new Array(outputSize);
    const std = Math.sqrt(2.0 / inputSize);

    for (let i = 0; i < outputSize; i++) {
        weights[i] = new Array(inputSize);
        for (let j = 0; j < inputSize; j++) {
            const u1 = Math.random();
            const u2 = Math.random();
            const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            weights[i][j] = z * std;
        }
    }
    return weights;
}

/**
 * ReLU activation
 */
function relu(x) {
    return x > 0 ? x : 0;
}

/**
 * ReLU derivative
 */
function reluDerivative(x) {
    return x > 0 ? 1 : 0;
}

async function main() {
    console.log('\n' + '='.repeat(60));
    console.log('TRAINING - MODEL 1 (2-Layer Network)');
    console.log('='.repeat(60));
    console.log('\nConfiguration:');
    console.log(`  Embedding dim:    ${CONFIG.embeddingDim}`);
    console.log(`  Hidden dim:       ${CONFIG.hiddenDim}`);
    console.log(`  Context window:   ${CONFIG.contextWindow}`);
    console.log(`  Learning rate:    ${CONFIG.learningRate}`);
    console.log(`  L2 lambda:        ${CONFIG.l2Lambda}`);
    console.log(`  Epochs:           ${CONFIG.epochs}`);

    const startTime = Date.now();

    if (!fs.existsSync(CONFIG.checkpointDir)) {
        fs.mkdirSync(CONFIG.checkpointDir, { recursive: true });
    }

    // ============================================
    // LOAD OR INITIALIZE
    // ============================================
    let embeddings = null;
    let wordToId = null;
    let idToWord = null;
    let vocabSize = 0;
    let W1 = null;
    let W2 = null;
    let startEpoch = 0;
    let startBatch = 0;

    const checkpointFiles = fs.existsSync(CONFIG.checkpointDir)
        ? fs.readdirSync(CONFIG.checkpointDir).filter(f => f.startsWith('checkpoint_epoch_'))
        : [];

    if (checkpointFiles.length > 0) {
        checkpointFiles.sort();
        const latestCheckpoint = checkpointFiles[checkpointFiles.length - 1];
        console.log(`\nResuming from checkpoint: ${latestCheckpoint}`);

        const checkpointPath = path.join(CONFIG.checkpointDir, latestCheckpoint);
        const checkpoint = JSON.parse(fs.readFileSync(checkpointPath, 'utf8'));

        embeddings = checkpoint.embeddings;
        wordToId = new Map(checkpoint.wordToId);
        idToWord = checkpoint.idToWord;
        vocabSize = checkpoint.vocabSize;
        W1 = checkpoint.W1;
        W2 = checkpoint.W2;
        startEpoch = checkpoint.epoch;
        startBatch = checkpoint.batch + 1;

        console.log(`  Epoch: ${startEpoch}, Batch: ${startBatch}`);
    } else {
        console.log(`\nLoading dictionary from: ${CONFIG.dictionaryFile}`);

        const lines = fs.readFileSync(CONFIG.dictionaryFile, 'utf8').split('\n');
        vocabSize = 0;
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].trim().length > 0) vocabSize++;
        }

        console.log(`  Vocabulary size: ${vocabSize}`);

        embeddings = new Array(vocabSize);
        wordToId = new Map();
        idToWord = new Array(vocabSize);

        const embDim = CONFIG.embeddingDim;
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line.length === 0) continue;

            const parts = line.split('\t');
            const word = parts[0];
            const id = parseInt(parts[1]);
            const vectorStr = parts[3];

            const vectorParts = vectorStr.split(' ');
            const vector = new Array(embDim);
            for (let j = 0; j < embDim; j++) {
                vector[j] = parseFloat(vectorParts[j]);
            }

            embeddings[id] = vector;
            wordToId.set(word, id);
            idToWord[id] = word;
        }

        console.log(`  Embeddings loaded: ${embeddings.length}`);
        console.log('\nInitializing network weights...');

        W1 = initializeWeights(CONFIG.embeddingDim, CONFIG.hiddenDim);
        W2 = initializeWeights(CONFIG.hiddenDim, CONFIG.embeddingDim);

        console.log(`  W1: ${CONFIG.embeddingDim} → ${CONFIG.hiddenDim}`);
        console.log(`  W2: ${CONFIG.hiddenDim} → ${CONFIG.embeddingDim}`);
    }

    // ============================================
    // LOAD TRAINING DATA
    // ============================================
    console.log(`\nLoading training data from: ${CONFIG.parquetDir}`);

    const parquetFiles = fs.readdirSync(CONFIG.parquetDir)
        .filter(f => f.endsWith('.parquet'))
        .sort();

    console.log(`Found ${parquetFiles.length} parquet files`);

    const allTexts = [];
    const numFiles = parquetFiles.length;
    for (let i = 0; i < numFiles; i++) {
        const filepath = path.join(CONFIG.parquetDir, parquetFiles[i]);
        console.log(`  [${i + 1}/${numFiles}] Reading ${parquetFiles[i]}...`);

        try {
            const pythonCmd = `python scripts/read_parquet.py "${filepath}"`;
            const output = execSync(pythonCmd, {
                encoding: 'utf8',
                maxBuffer: 500 * 1024 * 1024
            });
            const documents = JSON.parse(output);
            allTexts.push(...documents);
        } catch (error) {
            console.error(`Error reading ${filepath}:`, error.message);
        }
    }

    console.log(`Loaded ${allTexts.length} documents`);

    // ============================================
    // TOKENIZE DOCUMENTS
    // ============================================
    console.log('\nTokenizing documents...');

    const tokenizedDocs = [];
    const numDocs = allTexts.length;

    for (let docIdx = 0; docIdx < numDocs; docIdx++) {
        const text = allTexts[docIdx];
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

        if ((docIdx + 1) % 10000 === 0) {
            console.log(`  Tokenized ${docIdx + 1}/${numDocs} documents`);
        }
    }

    console.log(`Tokenized ${tokenizedDocs.length} documents`);

    // ============================================
    // TRAINING LOOP
    // ============================================
    console.log('\n' + '='.repeat(60));
    console.log('STARTING TRAINING');
    console.log('='.repeat(60));

    const contextSize = CONFIG.contextWindow;
    const embDim = CONFIG.embeddingDim;
    const hiddenDim = CONFIG.hiddenDim;
    const lr = CONFIG.learningRate;

    // Position weights
    const positionWeights = new Array(contextSize * 2 + 1);
    positionWeights[contextSize] = 0;
    for (let i = 0; i < contextSize; i++) {
        const weight = 1.0 / (contextSize - i);
        positionWeights[i] = weight;
        positionWeights[contextSize * 2 - i] = weight;
    }

    let batchCount = startBatch;
    const numTokenizedDocs = tokenizedDocs.length;
    const batchSize = CONFIG.batchSize;

    // Initialize gradient accumulators
    const gradW1 = new Array(hiddenDim);
    const gradW2 = new Array(embDim);
    for (let i = 0; i < hiddenDim; i++) {
        gradW1[i] = new Array(embDim).fill(0);
    }
    for (let i = 0; i < embDim; i++) {
        gradW2[i] = new Array(hiddenDim).fill(0);
    }
    const gradEmbeddings = new Map();
    let batchWindowCount = 0;

    // Timing for checkpoint intervals
    let lastCheckpointTime = Date.now();

    for (let epoch = startEpoch; epoch < CONFIG.epochs; epoch++) {
        console.log(`\nEpoch ${epoch + 1}/${CONFIG.epochs}`);
        let epochLoss = 0;
        let windowsProcessed = 0;

        for (let docIdx = 0; docIdx < numTokenizedDocs; docIdx++) {
            const tokens = tokenizedDocs[docIdx];
            const numTokens = tokens.length;

            for (let center = contextSize; center < numTokens - contextSize; center++) {
                const targetId = tokens[center];

                // ============================================
                // FORWARD PASS
                // ============================================

                // 1. Compute weighted average of context
                const avgContext = new Array(embDim);
                for (let d = 0; d < embDim; d++) {
                    avgContext[d] = 0;
                }

                let totalWeight = 0;
                for (let offset = -contextSize; offset <= contextSize; offset++) {
                    if (offset === 0) continue;

                    const contextId = tokens[center + offset];
                    const weight = positionWeights[offset + contextSize];
                    const contextVec = embeddings[contextId];

                    for (let d = 0; d < embDim; d++) {
                        avgContext[d] += contextVec[d] * weight;
                    }
                    totalWeight += weight;
                }

                for (let d = 0; d < embDim; d++) {
                    avgContext[d] /= totalWeight;
                }

                // 2. Hidden Layer 1: h1 = ReLU(W1 * avgContext)
                const h1_pre = new Array(hiddenDim);
                const h1 = new Array(hiddenDim);

                for (let i = 0; i < hiddenDim; i++) {
                    h1_pre[i] = 0;
                    for (let j = 0; j < embDim; j++) {
                        h1_pre[i] += W1[i][j] * avgContext[j];
                    }
                    h1[i] = relu(h1_pre[i]);
                }

                // 3. Output Layer: output = W2 * h1
                const predicted = new Array(embDim);

                for (let i = 0; i < embDim; i++) {
                    predicted[i] = 0;
                    for (let j = 0; j < hiddenDim; j++) {
                        predicted[i] += W2[i][j] * h1[j];
                    }
                }

                // ============================================
                // COMPUTE LOSS (Cosine Similarity)
                // ============================================
                const targetVec = embeddings[targetId];

                // Compute cosine similarity: dot / (||predicted|| * ||target||)
                let dot = 0;
                let predNorm = 0;
                let targetNorm = 0;

                for (let d = 0; d < embDim; d++) {
                    dot += predicted[d] * targetVec[d];
                    predNorm += predicted[d] * predicted[d];
                    targetNorm += targetVec[d] * targetVec[d];
                }

                predNorm = Math.sqrt(predNorm);
                targetNorm = Math.sqrt(targetNorm);

                const cosineSim = dot / (predNorm * targetNorm + 1e-8);
                const loss = 1 - cosineSim;  // Loss: 0 when identical, 2 when opposite

                epochLoss += loss;
                windowsProcessed++;
                batchCount++;

                // ============================================
                // BACKWARD PASS (Cosine Similarity Gradient)
                // ============================================

                // Gradient of cosine similarity loss
                // dL/dPred = -1 * d(cosineSim)/dPred
                // d(cosineSim)/dPred = (target/(||pred||*||target||)) - (pred*dot)/(||pred||^3 * ||target||)

                const dOutput = new Array(embDim);
                const denom = predNorm * targetNorm + 1e-8;

                for (let d = 0; d < embDim; d++) {
                    const term1 = targetVec[d] / denom;
                    const term2 = (predicted[d] * dot) / (predNorm * predNorm * denom + 1e-8);
                    dOutput[d] = -(term1 - term2);  // Negative because we minimize (1 - cosineSim)
                }

                // Accumulate W2 gradients: dL/dW2 = dOutput ⊗ h1^T
                for (let i = 0; i < embDim; i++) {
                    for (let j = 0; j < hiddenDim; j++) {
                        gradW2[i][j] += dOutput[i] * h1[j];
                    }
                }

                // Gradient at h1: dL/dh1 = W2^T * dOutput
                const dH1 = new Array(hiddenDim);
                for (let j = 0; j < hiddenDim; j++) {
                    dH1[j] = 0;
                    for (let i = 0; i < embDim; i++) {
                        dH1[j] += W2[i][j] * dOutput[i];
                    }
                    // Apply ReLU derivative
                    dH1[j] *= reluDerivative(h1_pre[j]);
                }

                // Accumulate W1 gradients: dL/dW1 = dH1 ⊗ avgContext^T
                for (let i = 0; i < hiddenDim; i++) {
                    for (let j = 0; j < embDim; j++) {
                        gradW1[i][j] += dH1[i] * avgContext[j];
                    }
                }

                // Accumulate target embedding gradient
                if (!gradEmbeddings.has(targetId)) {
                    gradEmbeddings.set(targetId, new Array(embDim).fill(0));
                }
                const targetGrad = gradEmbeddings.get(targetId);
                for (let d = 0; d < embDim; d++) {
                    targetGrad[d] += -dOutput[d];
                }

                batchWindowCount++;

                // Apply batch update when batch is full
                if (batchWindowCount === batchSize) {
                    // Update W1
                    for (let i = 0; i < hiddenDim; i++) {
                        for (let j = 0; j < embDim; j++) {
                            W1[i][j] -= lr * (gradW1[i][j] / batchSize);
                            gradW1[i][j] = 0;
                        }
                    }

                    // Update W2
                    for (let i = 0; i < embDim; i++) {
                        for (let j = 0; j < hiddenDim; j++) {
                            W2[i][j] -= lr * (gradW2[i][j] / batchSize);
                            gradW2[i][j] = 0;
                        }
                    }

                    // Update embeddings and normalize to unit length
                    for (const [id, grad] of gradEmbeddings.entries()) {
                        const vec = embeddings[id];

                        // Apply gradient update
                        for (let d = 0; d < embDim; d++) {
                            vec[d] -= lr * (grad[d] / batchSize);
                        }

                        // L2 normalize to unit length (prevent collapse)
                        let norm = 0;
                        for (let d = 0; d < embDim; d++) {
                            norm += vec[d] * vec[d];
                        }
                        norm = Math.sqrt(norm) + 1e-8;
                        for (let d = 0; d < embDim; d++) {
                            vec[d] /= norm;
                        }
                    }
                    gradEmbeddings.clear();
                    batchWindowCount = 0;
                }

                // Checkpoint
                if (batchCount % CONFIG.checkpointEvery === 0) {
                    const avgLoss = epochLoss / windowsProcessed;
                    const currentTime = Date.now();
                    const timeSinceLastCheckpoint = (currentTime - lastCheckpointTime) / 1000;
                    const windowsPerSecond = Math.floor(CONFIG.checkpointEvery / timeSinceLastCheckpoint);

                    console.log('\n' + '='.repeat(60));
                    console.log(`CHECKPOINT at batch ${batchCount}`);
                    console.log('='.repeat(60));
                    console.log(`Epoch: ${epoch + 1}/${CONFIG.epochs}`);
                    console.log(`Windows: ${windowsProcessed}`);
                    console.log(`Avg Loss: ${avgLoss.toFixed(6)}`);
                    console.log(`Time for last ${CONFIG.checkpointEvery} batches: ${timeSinceLastCheckpoint.toFixed(1)}s (${windowsPerSecond} windows/s)`);

                    const checkpoint = {
                        epoch: epoch,
                        batch: batchCount,
                        embeddings: embeddings,
                        wordToId: Array.from(wordToId.entries()),
                        idToWord: idToWord,
                        vocabSize: vocabSize,
                        W1: W1,
                        W2: W2,
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

            if ((docIdx + 1) % 5000 === 0) {
                console.log(`  Processed ${docIdx + 1}/${numTokenizedDocs} documents`);
            }
        }

        // Apply any remaining gradients at end of epoch
        if (batchWindowCount > 0) {
            // Update W1
            for (let i = 0; i < hiddenDim; i++) {
                for (let j = 0; j < embDim; j++) {
                    W1[i][j] -= lr * (gradW1[i][j] / batchWindowCount);
                    gradW1[i][j] = 0;
                }
            }

            // Update W2
            for (let i = 0; i < embDim; i++) {
                for (let j = 0; j < hiddenDim; j++) {
                    W2[i][j] -= lr * (gradW2[i][j] / batchWindowCount);
                    gradW2[i][j] = 0;
                }
            }

            // Update embeddings and normalize
            for (const [id, grad] of gradEmbeddings.entries()) {
                const vec = embeddings[id];

                // Apply gradient update
                for (let d = 0; d < embDim; d++) {
                    vec[d] -= lr * (grad[d] / batchWindowCount);
                }

                // L2 normalize to unit length
                let norm = 0;
                for (let d = 0; d < embDim; d++) {
                    norm += vec[d] * vec[d];
                }
                norm = Math.sqrt(norm) + 1e-8;
                for (let d = 0; d < embDim; d++) {
                    vec[d] /= norm;
                }
            }
            gradEmbeddings.clear();
            batchWindowCount = 0;
        }

        const avgEpochLoss = epochLoss / windowsProcessed;
        console.log(`\nEpoch ${epoch + 1} complete | Avg Loss: ${avgEpochLoss.toFixed(6)}`);
    }

    // ============================================
    // SAVE FINAL MODEL
    // ============================================
    console.log('\nSaving final model...');

    const finalPath = './data/model_final.ndjson';
    const writeStream = fs.createWriteStream(finalPath);

    const precision = 4;
    for (let i = 0; i < vocabSize; i++) {
        const word = idToWord[i];
        const vec = embeddings[i];

        let vectorStr = '';
        const vecLen = vec.length;
        for (let j = 0; j < vecLen; j++) {
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

    console.log(`Final model saved to: ${finalPath}`);

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    console.log('\n' + '='.repeat(60));
    console.log('TRAINING COMPLETE!');
    console.log('='.repeat(60));
    console.log(`Time elapsed: ${elapsed}s`);
    console.log('');
}

main().catch(error => {
    console.error('\n❌ Training failed:', error.message);
    console.error(error.stack);
    process.exit(1);
});
