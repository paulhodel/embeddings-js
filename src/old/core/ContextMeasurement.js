/**
 * ContextMeasurement - Represents a context as a measurement operator
 *
 * In CSS, each context C acts as a spectral measurement pattern that
 * probes specific frequencies of a word's hidden spectrum.
 *
 * Context ≈ a noisy, partial observation of S_w
 */

export class ContextMeasurement {
    /**
     * @param {number} frequencyDim - Total frequency dimension space
     * @param {number} windowSize - Context window size
     * @param {number} negativeCount - Number of negative samples per positive
     * @param {number} vocabSize - Total vocabulary size (for negative sampling)
     */
    constructor(frequencyDim = 100, windowSize = 2, negativeCount = 5, vocabSize = 1000) {
        this.frequencyDim = frequencyDim;
        this.windowSize = windowSize;
        this.negativeCount = negativeCount;
        this.vocabSize = vocabSize;

        // Store aggregated context patterns for each word
        // Map: wordId -> Array of measurement patterns
        this.measurements = new Map();
    }

    /**
     * Create a measurement pattern from context words
     * The context is represented as a sparse combination of context word spectra
     *
     * SIMPLIFIED: Now real-valued only (no complex numbers)
     *
     * @param {Array<number>} contextWordIds - IDs of context words
     * @param {SpectralWord} spectralWord - SpectralWord instance
     * @returns {Array<number>} - Dense measurement pattern (real-valued)
     */
    createMeasurementPattern(contextWordIds, spectralWord) {
        const pattern = new Array(this.frequencyDim).fill(0);

        if (contextWordIds.length === 0) {
            return pattern;
        }

        // Aggregate context word spectra into a single measurement pattern
        for (const contextId of contextWordIds) {
            const contextVector = spectralWord.toDenseVector(contextId);

            // Add context word's spectrum to the pattern
            for (let i = 0; i < contextVector.length; i++) {
                pattern[i] += contextVector[i];
            }
        }

        // Normalize by number of context words
        const norm = contextWordIds.length;
        for (let i = 0; i < pattern.length; i++) {
            pattern[i] /= norm;
        }

        // Add noise to simulate partial/noisy measurement
        const noiseLevel = 0.05;
        for (let i = 0; i < pattern.length; i++) {
            pattern[i] += (Math.random() - 0.5) * noiseLevel;
        }

        return pattern;
    }

    /**
     * Extract context windows from a sequence of word IDs
     * Returns pairs of (targetWord, contextWords)
     *
     * @param {Array<number>} wordIds - Sequence of word IDs
     * @returns {Array<{target: number, context: Array<number>}>}
     */
    extractContextWindows(wordIds) {
        const windows = [];

        for (let i = 0; i < wordIds.length; i++) {
            const target = wordIds[i];
            const context = [];

            // Get words within window
            for (let j = Math.max(0, i - this.windowSize); j <= Math.min(wordIds.length - 1, i + this.windowSize); j++) {
                if (j !== i) { // Exclude the target word itself
                    context.push(wordIds[j]);
                }
            }

            if (context.length > 0) {
                windows.push({target, context});
            }
        }

        return windows;
    }

    /**
     * Record a measurement for a word given its context
     * This stores the measurement pattern associated with this context
     *
     * @param {number} targetWordId
     * @param {Array<number>} pattern - The measurement pattern
     */
    recordMeasurement(targetWordId, pattern) {
        if (!this.measurements.has(targetWordId)) {
            this.measurements.set(targetWordId, []);
        }
        this.measurements.get(targetWordId).push(pattern);
    }

    /**
     * Get all measurements for a word
     */
    getMeasurements(wordId) {
        return this.measurements.get(wordId) || [];
    }

    /**
     * Get the number of measurements (contexts) observed for a word
     */
    getMeasurementCount(wordId) {
        return this.getMeasurements(wordId).length;
    }

    /**
     * Clear measurements (for batch processing)
     */
    clearMeasurements() {
        this.measurements.clear();
    }

    /**
     * Sample negative words for contrastive learning
     * Randomly select words from vocabulary that are not the target
     *
     * @param {number} targetWordId - The positive target word to exclude
     * @param {number} count - Number of negative samples
     * @returns {Array<number>} - Array of negative word IDs
     */
    sampleNegativeWords(targetWordId, count = null) {
        if (count === null) {
            count = this.negativeCount;
        }

        const negatives = [];
        const maxAttempts = count * 10; // Prevent infinite loop
        let attempts = 0;

        while (negatives.length < count && attempts < maxAttempts) {
            const randomId = Math.floor(Math.random() * this.vocabSize);

            // Don't use the target word or duplicates
            if (randomId !== targetWordId && !negatives.includes(randomId)) {
                negatives.push(randomId);
            }

            attempts++;
        }

        return negatives;
    }

    /**
     * Calculate contrastive loss for a word
     *
     * Contrastive objective:
     * - Positive score (target word with its context) should be HIGH
     * - Negative scores (random words with context) should be LOW
     *
     * Loss = Σ_contexts [max(0, margin - pos_score + neg_score)]
     *
     * @param {number} wordId
     * @param {SpectralWord} spectralWord
     * @param {number} margin - Margin for contrastive loss
     * @returns {number} - Average contrastive loss
     */
    calculateReconstructionLoss(wordId, spectralWord, margin = 0.5) {
        const measurements = this.getMeasurements(wordId);

        if (measurements.length === 0) {
            return 0;
        }

        let totalLoss = 0;
        let numComparisons = 0;

        for (const pattern of measurements) {
            // Positive score: target word with its actual context
            const posScore = spectralWord.measure(wordId, pattern);

            // Sample negative words
            const negativeWords = this.sampleNegativeWords(wordId);

            // Negative scores: random words with this context
            for (const negWordId of negativeWords) {
                const negScore = spectralWord.measure(negWordId, pattern);

                // Contrastive loss: push positive up, negative down
                // Loss increases if: negative score too close to positive score
                const loss = Math.max(0, margin - posScore + negScore);
                totalLoss += loss;
                numComparisons++;
            }
        }

        return numComparisons > 0 ? totalLoss / numComparisons : 0;
    }

    /**
     * Calculate gradient of contrastive loss w.r.t. word spectrum
     * Returns gradient in dense real-valued vector form
     *
     * SIMPLIFIED: Real-valued gradient (no complex math)
     *
     * Gradient directions:
     * - For target word: gradient pushes spectrum toward context (if loss > 0)
     * - For negative words: gradient pushes spectrum away from context (if loss > 0)
     *
     * @param {number} wordId
     * @param {SpectralWord} spectralWord
     * @param {number} margin - Margin for contrastive loss
     * @returns {Array<number>} - Gradient vector (real-valued)
     */
    calculateGradient(wordId, spectralWord, margin = 0.5) {
        const measurements = this.getMeasurements(wordId);
        const gradient = new Array(this.frequencyDim).fill(0);

        if (measurements.length === 0) {
            return gradient;
        }

        let numUpdates = 0;

        for (const pattern of measurements) {
            // Positive score
            const posScore = spectralWord.measure(wordId, pattern);

            // Sample negative words
            const negativeWords = this.sampleNegativeWords(wordId);

            for (const negWordId of negativeWords) {
                const negScore = spectralWord.measure(negWordId, pattern);

                // Only update if loss is active (within margin)
                const loss = margin - posScore + negScore;

                if (loss > 0) {
                    // Gradient pushes positive word toward context pattern
                    // (increase alignment with context)
                    for (let i = 0; i < pattern.length; i++) {
                        gradient[i] -= pattern[i]; // Negative gradient = increase score
                    }

                    numUpdates++;
                }
            }
        }

        // Normalize gradient
        if (numUpdates > 0) {
            for (let i = 0; i < gradient.length; i++) {
                gradient[i] /= numUpdates;
            }
        }

        return gradient;
    }

    /**
     * Calculate gradient for a negative word
     * This pushes the negative word's spectrum AWAY from the context
     *
     * SIMPLIFIED: Real-valued gradient (no complex math)
     *
     * @param {number} negWordId - Negative word ID
     * @param {number} posWordId - Positive word ID (to get its contexts)
     * @param {SpectralWord} spectralWord
     * @param {number} margin - Margin for contrastive loss
     * @returns {Array<number>} - Gradient vector (real-valued)
     */
    calculateNegativeGradient(negWordId, posWordId, spectralWord, margin = 0.5) {
        const measurements = this.getMeasurements(posWordId);
        const gradient = new Array(this.frequencyDim).fill(0);

        if (measurements.length === 0) {
            return gradient;
        }

        let numUpdates = 0;

        for (const pattern of measurements) {
            const posScore = spectralWord.measure(posWordId, pattern);
            const negScore = spectralWord.measure(negWordId, pattern);

            const loss = margin - posScore + negScore;

            if (loss > 0) {
                // Gradient pushes negative word AWAY from context pattern
                // (decrease alignment with context)
                for (let i = 0; i < pattern.length; i++) {
                    gradient[i] += pattern[i]; // Positive gradient = decrease score
                }

                numUpdates++;
            }
        }

        // Normalize gradient
        if (numUpdates > 0) {
            for (let i = 0; i < gradient.length; i++) {
                gradient[i] /= numUpdates;
            }
        }

        return gradient;
    }
}
