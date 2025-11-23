/**
 * CSSTrainer - Compressive Semantic Spectroscopy Trainer
 *
 * Implements the sparse inverse problem:
 * Find the sparsest spectrum S_w that explains all contextual measurements
 *
 * Training becomes a global reconstruction problem, not local prediction.
 */

import {SpectralWord} from './SpectralWord.js';
import {ContextMeasurement} from './ContextMeasurement.js';

export class CSSTrainer {
    /**
     * @param {Object} config - Training configuration
     */
    constructor(config = {}) {
        this.config = {
            frequencyDim: config.frequencyDim || 100,
            maxFrequencies: config.maxFrequencies || 5,
            windowSize: config.windowSize || 2,
            learningRate: config.learningRate || 0.01,
            sparsityPenalty: config.sparsityPenalty || 0.001,
            epochs: config.epochs || 10,
            batchSize: config.batchSize || 100,
            negativeCount: config.negativeCount || 5,
            margin: config.margin || 0.5,
            updateNegatives: config.updateNegatives !== false, // Default true
            ...config
        };

        this.spectralWord = null;
        this.contextMeasurement = null;
        this.vocabSize = 0;
    }

    /**
     * Initialize the model with vocabulary size
     */
    initialize(vocabSize) {
        this.vocabSize = vocabSize;
        this.spectralWord = new SpectralWord(
            vocabSize,
            this.config.maxFrequencies,
            this.config.frequencyDim
        );
        this.contextMeasurement = new ContextMeasurement(
            this.config.frequencyDim,
            this.config.windowSize,
            this.config.negativeCount,
            vocabSize
        );

        console.log(`Initialized CSS model:`);
        console.log(`  Vocabulary size: ${vocabSize}`);
        console.log(`  Frequency dimension: ${this.config.frequencyDim}`);
        console.log(`  Max frequencies per word: ${this.config.maxFrequencies}`);
        console.log(`  Negative samples per positive: ${this.config.negativeCount}`);
    }

    /**
     * Phase 1: Collect all contextual measurements
     * Each (word, context) pair is a measurement observation
     */
    collectMeasurements(corpus) {
        console.log('\n=== Phase 1: Collecting Measurements ===');

        this.contextMeasurement.clearMeasurements();
        let totalMeasurements = 0;

        for (let docIdx = 0; docIdx < corpus.length; docIdx++) {
            const wordIds = corpus[docIdx];

            if (docIdx % 1000 === 0 && docIdx > 0) {
                console.log(`  Processed ${docIdx}/${corpus.length} documents...`);
            }

            // Extract context windows
            const windows = this.contextMeasurement.extractContextWindows(wordIds);

            // Create measurement patterns
            for (const window of windows) {
                const pattern = this.contextMeasurement.createMeasurementPattern(
                    window.context,
                    this.spectralWord
                );
                this.contextMeasurement.recordMeasurement(window.target, pattern);
                totalMeasurements++;
            }
        }

        console.log(`  Total measurements collected: ${totalMeasurements}`);

        // Report measurement statistics
        const wordsWithMeasurements = Array.from(this.contextMeasurement.measurements.keys()).length;
        console.log(`  Words with measurements: ${wordsWithMeasurements}`);
    }

    /**
     * Phase 2: Sparse reconstruction via gradient descent with contrastive learning
     * Solve the inverse problem: find sparse spectra that explain measurements
     *
     * Now includes negative sampling:
     * - Positive words: pushed toward their contexts
     * - Negative words: pushed away from unrelated contexts
     *
     * @param {Function} epochCallback - Optional callback called after each epoch: (epoch, stats) => void
     */
    sparseReconstruction(epochCallback = null) {
        console.log('\n=== Phase 2: Sparse Reconstruction (with TWO-PHASE PRUNING) ===');

        const wordIds = Array.from(this.contextMeasurement.measurements.keys());
        let globalUpdateStep = 0; // Track global update steps for pruning schedule

        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            let epochLoss = 0;
            let epochSparsity = 0;
            let processedWords = 0;
            let negativeUpdates = 0;
            let frequenciesAdded = 0;
            let frequenciesPruned = 0;

            // Training progress for pruning schedule
            const progress = epoch / this.config.epochs;
            const phase = progress < 0.2 ? 'EXPLORATION' : 'REFINEMENT';

            console.log(`  Epoch ${epoch + 1}/${this.config.epochs} (${phase} phase, ${(progress * 100).toFixed(0)}% complete)`);

            // Shuffle word order each epoch
            this.shuffleArray(wordIds);

            // Track negative words to update in this epoch
            const negativeWordsToUpdate = new Map(); // negWordId -> [posWordIds]

            // Process in batches
            for (let i = 0; i < wordIds.length; i += this.config.batchSize) {
                const batch = wordIds.slice(i, Math.min(i + this.config.batchSize, wordIds.length));

                for (const wordId of batch) {
                    const beforeSparsity = this.spectralWord.getSparsity(wordId);

                    // Calculate contrastive loss (includes negative sampling)
                    const loss = this.contextMeasurement.calculateReconstructionLoss(
                        wordId,
                        this.spectralWord,
                        this.config.margin
                    );

                    // Calculate gradient for positive word
                    const gradient = this.contextMeasurement.calculateGradient(
                        wordId,
                        this.spectralWord,
                        this.config.margin
                    );

                    // Update positive word spectrum with pruning config
                    this.spectralWord.updateSpectrum(
                        wordId,
                        gradient,
                        this.config.learningRate,
                        this.config.sparsityPenalty,
                        {
                            epoch,
                            maxEpochs: this.config.epochs,
                            updateStep: globalUpdateStep,
                            allowAddFreq: true
                        }
                    );

                    // Normalize spectrum to prevent runaway amplitudes
                    this.spectralWord.normalizeSpectrum(wordId, 'euclidean');

                    const afterSparsity = this.spectralWord.getSparsity(wordId);
                    if (afterSparsity > beforeSparsity) frequenciesAdded++;
                    if (afterSparsity < beforeSparsity) frequenciesPruned++;

                    globalUpdateStep++;

                    // Collect negative words for update
                    if (this.config.updateNegatives) {
                        const measurements = this.contextMeasurement.getMeasurements(wordId);
                        if (measurements.length > 0) {
                            // Sample some negatives for this positive word
                            const negatives = this.contextMeasurement.sampleNegativeWords(wordId);

                            for (const negId of negatives) {
                                if (!negativeWordsToUpdate.has(negId)) {
                                    negativeWordsToUpdate.set(negId, []);
                                }
                                negativeWordsToUpdate.get(negId).push(wordId);
                            }
                        }
                    }

                    // Track metrics
                    epochLoss += loss;
                    epochSparsity += this.spectralWord.getSparsity(wordId);
                    processedWords++;
                }
            }

            // Update negative words
            if (this.config.updateNegatives) {
                for (const [negWordId, posWordIds] of negativeWordsToUpdate.entries()) {
                    // Accumulate gradients from all contexts this negative appeared in
                    const accumulatedGradient = new Array(this.config.frequencyDim).fill(0);

                    for (const posWordId of posWordIds) {
                        const negGradient = this.contextMeasurement.calculateNegativeGradient(
                            negWordId,
                            posWordId,
                            this.spectralWord,
                            this.config.margin
                        );

                        // Accumulate
                        for (let j = 0; j < negGradient.length; j++) {
                            accumulatedGradient[j] += negGradient[j];
                        }
                    }

                    // Normalize and update
                    for (let j = 0; j < accumulatedGradient.length; j++) {
                        accumulatedGradient[j] /= posWordIds.length;
                    }

                    this.spectralWord.updateSpectrum(
                        negWordId,
                        accumulatedGradient,
                        this.config.learningRate * 0.1, // Smaller LR for negatives
                        this.config.sparsityPenalty,
                        {
                            epoch,
                            maxEpochs: this.config.epochs,
                            updateStep: globalUpdateStep,
                            allowAddFreq: false // Don't add frequencies for negatives
                        }
                    );

                    // Normalize negative word spectrum
                    this.spectralWord.normalizeSpectrum(negWordId, 'euclidean');

                    globalUpdateStep++;
                    negativeUpdates++;
                }
            }

            // Report epoch statistics
            const avgLoss = epochLoss / processedWords;
            const avgSparsity = epochSparsity / processedWords;

            console.log(`    Avg Loss: ${avgLoss.toFixed(6)}`);
            console.log(`    Avg Sparsity: ${avgSparsity.toFixed(2)} active frequencies`);
            if (progress < 0.2) {
                console.log(`    Frequencies added: ${frequenciesAdded}`);
                console.log(`    Frequencies pruned: ${frequenciesPruned}`);
            }
            if (this.config.updateNegatives) {
                console.log(`    Negative updates: ${negativeUpdates}`);
            }

            // Decay learning rate
            if ((epoch + 1) % 5 === 0) {
                this.config.learningRate *= 0.9;
                console.log(`    Learning rate decayed to ${this.config.learningRate.toFixed(6)}`);
            }

            // Call epoch callback if provided (for snapshots, logging, etc.)
            if (epochCallback) {
                epochCallback(epoch, {
                    avgLoss,
                    avgSparsity,
                    frequenciesAdded,
                    frequenciesPruned,
                    negativeUpdates,
                    phase
                });
            }
        }
    }

    /**
     * Train on corpus
     * @param {Array<Array<number>>} corpus - Array of documents (each document is array of word IDs)
     * @param {Function} epochCallback - Optional callback after each epoch: (epoch, stats) => void
     */
    train(corpus, epochCallback = null) {
        console.log('\n========================================');
        console.log('COMPRESSIVE SEMANTIC SPECTRA');
        console.log('Training as Sparse Inverse Problem');
        console.log('========================================');

        const startTime = Date.now();

        // Phase 1: Collect measurements
        this.collectMeasurements(corpus);

        // Phase 2: Sparse reconstruction
        this.sparseReconstruction(epochCallback);

        const duration = ((Date.now() - startTime) / 1000).toFixed(2);
        console.log(`\nTraining completed in ${duration}s`);
    }

    /**
     * Get word spectrum for interpretation
     */
    getWordSpectrum(wordId, topK = 3) {
        return this.spectralWord.getDominantFrequencies(wordId, topK);
    }

    /**
     * Calculate spectral similarity between two words
     * Based on overlap in their active frequencies
     *
     * SIMPLIFIED: Real-valued dot product (no complex math)
     */
    spectralSimilarity(wordId1, wordId2) {
        // Convert to dense for comparison
        const vec1 = this.spectralWord.toDenseVector(wordId1);
        const vec2 = this.spectralWord.toDenseVector(wordId2);

        // Simple dot product (real-valued)
        let dotProduct = 0;

        for (let i = 0; i < this.config.frequencyDim; i++) {
            dotProduct += vec1[i] * vec2[i];
        }

        // Normalize by magnitudes (cosine similarity)
        const norm1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
        const norm2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));

        if (norm1 === 0 || norm2 === 0) return 0;

        return dotProduct / (norm1 * norm2);
    }

    /**
     * Find most similar words to a target word based on spectral overlap
     */
    findSimilar(wordId, topK = 5) {
        const similarities = [];

        for (let otherWordId = 0; otherWordId < this.vocabSize; otherWordId++) {
            if (otherWordId === wordId) continue;

            const sim = this.spectralSimilarity(wordId, otherWordId);
            similarities.push({wordId: otherWordId, similarity: sim});
        }

        // Sort by similarity
        similarities.sort((a, b) => b.similarity - a.similarity);

        return similarities.slice(0, topK);
    }

    /**
     * Export trained model
     */
    exportModel() {
        const model = {
            config: this.config,
            vocabSize: this.vocabSize,
            spectra: {}
        };

        // Export all word spectra
        for (const [wordId, spectrum] of this.spectralWord.spectra.entries()) {
            model.spectra[wordId] = spectrum;
        }

        return model;
    }

    /**
     * Import trained model
     */
    importModel(model) {
        this.config = model.config;
        this.vocabSize = model.vocabSize;

        this.initialize(model.vocabSize);

        // Import spectra
        for (const [wordId, spectrum] of Object.entries(model.spectra)) {
            this.spectralWord.spectra.set(parseInt(wordId), spectrum);
        }
    }

    /**
     * Utility: Shuffle array in place
     */
    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }
}
