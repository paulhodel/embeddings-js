/**
 * SpectralWord - Represents a word as a sparse spectrum
 *
 * SIMPLIFIED VERSION (no phase):
 * Each word has:
 * - A set of active frequencies (ω_k)
 * - Amplitudes (A_k) for each frequency
 *
 * S_w(ω) = Σ A_k * δ(ω - ω_k)
 *
 * This gives us:
 * - Polysemy as multiple peaks
 * - Context filtering
 * - Interpretable semantic modes
 * - Compact representation
 * - Simpler math and faster computation
 */

export class SpectralWord {
    /**
     * @param {number} vocabSize - Total vocabulary size
     * @param {number} maxFrequencies - Maximum number of frequencies (K)
     * @param {number} frequencyDim - Total frequency dimension space
     */
    constructor(vocabSize, maxFrequencies = 5, frequencyDim = 100) {
        this.vocabSize = vocabSize;
        this.maxFrequencies = maxFrequencies; // K - max active frequencies per word
        this.frequencyDim = frequencyDim; // Total frequency space dimension

        // Store sparse spectra for each word
        // Map: wordId -> { frequencies: [], amplitudes: [] }
        this.spectra = new Map();
    }

    /**
     * Initialize a word's spectrum randomly (DENSE start)
     *
     * Strategy: Start ALL words with maxFrequencies at TINY amplitudes
     * - Dense initialization allows exploration of many semantic directions
     * - Tiny amplitudes (0.0005-0.003) prevent early random signals from dominating
     * - Aggressive early pruning removes noise quickly
     * - Only reinforced frequencies survive
     *
     * This mirrors:
     * - Neural nets (dense random init)
     * - Sparse coding (dense code + later sparsification)
     * - k-SVD (dense initialization + iterative hard-thresholding)
     */
    initializeWord(wordId) {
        const frequencies = [];
        const amplitudes = [];

        // Sample random frequencies without replacement
        // Start DENSE (all words get maxFrequencies)
        const usedFreqs = new Set();
        for (let i = 0; i < this.maxFrequencies; i++) {
            let freq;
            do {
                freq = Math.floor(Math.random() * this.frequencyDim);
            } while (usedFreqs.has(freq));
            usedFreqs.add(freq);

            frequencies.push(freq);

            // CRITICAL: Tiny initial amplitudes [0.0005, 0.003]
            // This prevents early random context signals from dominating
            amplitudes.push(Math.random() * 0.0025 + 0.0005); // [0.0005, 0.003]
        }

        this.spectra.set(wordId, {frequencies, amplitudes});
    }

    /**
     * Get or initialize spectrum for a word
     */
    getSpectrum(wordId) {
        if (!this.spectra.has(wordId)) {
            this.initializeWord(wordId);
        }
        return this.spectra.get(wordId);
    }

    /**
     * Convert sparse spectrum to dense vector (for measurement)
     * Returns a real-valued vector (no complex numbers needed)
     */
    toDenseVector(wordId) {
        const spectrum = this.getSpectrum(wordId);
        const dense = new Array(this.frequencyDim).fill(0);

        for (let i = 0; i < spectrum.frequencies.length; i++) {
            const freq = spectrum.frequencies[i];
            const amp = spectrum.amplitudes[i];

            // Simple real-valued representation
            dense[freq] = amp;
        }

        return dense;
    }

    /**
     * Update a word's spectrum from gradient with TWO-PHASE PRUNING
     *
     * @param {number} wordId
     * @param {Array} gradient - gradient in dense form
     * @param {number} learningRate
     * @param {number} sparsityPenalty - L1 penalty for sparsity
     * @param {Object} pruningConfig - { epoch, maxEpochs, updateStep, allowAddFreq }
     *
     * TWO-PHASE PRUNING SCHEDULE:
     *
     * Phase 1 (0-20% of training): EXPLORATION
     * - Aggressive pruning (threshold ~0.001, 2-5x initial amplitude)
     * - Prune every step
     * - Allow adding new frequencies when gradient is strong
     * - Removes noise fast, lets meaningful signals emerge
     *
     * Phase 2 (20-100% of training): REFINEMENT
     * - Gentle pruning (threshold ~0.0001)
     * - Prune every 500-5000 steps
     * - No new frequencies added
     * - Stabilizes spectra for convergence
     */
    updateSpectrum(wordId, gradient, learningRate = 0.01, sparsityPenalty = 0.001, pruningConfig = {}) {
        const spectrum = this.getSpectrum(wordId);

        // Default pruning config
        const {
            epoch = 0,
            maxEpochs = 10,
            updateStep = 0,
            allowAddFreq = true
        } = pruningConfig;

        // Compute training progress (0 to 1)
        const progress = maxEpochs > 0 ? epoch / maxEpochs : 0;

        // Update existing frequencies
        for (let i = 0; i < spectrum.frequencies.length; i++) {
            const freq = spectrum.frequencies[i];
            const amp = spectrum.amplitudes[i];

            // Extract gradient for this frequency
            const grad = gradient[freq];

            // Update amplitude with gradient descent
            const newAmp = amp - learningRate * grad;

            // Apply sparsity penalty (soft thresholding on amplitude)
            spectrum.amplitudes[i] = Math.max(0, newAmp - sparsityPenalty);
        }

        // PHASE 1 (0-20%): EXPLORATION - Add new frequencies if gradient is strong
        if (progress < 0.2 && allowAddFreq && spectrum.frequencies.length < this.maxFrequencies) {
            // Find strongest gradient on unused frequencies
            const usedFreqs = new Set(spectrum.frequencies);
            let maxGrad = 0;
            let bestFreq = -1;

            for (let freq = 0; freq < this.frequencyDim; freq++) {
                if (!usedFreqs.has(freq)) {
                    const gradMag = Math.abs(gradient[freq]);
                    if (gradMag > maxGrad) {
                        maxGrad = gradMag;
                        bestFreq = freq;
                    }
                }
            }

            // Add frequency if gradient is strong enough (threshold: 0.1)
            // This allows model to discover important frequencies missed in initialization
            if (maxGrad > 0.1 && bestFreq !== -1) {
                spectrum.frequencies.push(bestFreq);
                spectrum.amplitudes.push(0.001); // Small initial amplitude
            }
        }

        // TWO-PHASE PRUNING
        let shouldPrune = false;
        let pruneThreshold = 0;

        if (progress < 0.2) {
            // PHASE 1 (0-20%): EXPLORATION
            // - Aggressive pruning every step
            // - Threshold: 0.001 (2-5x initial amplitude range)
            shouldPrune = true;
            pruneThreshold = 0.001;
        } else {
            // PHASE 2 (20-100%): REFINEMENT
            // - Gentle pruning every 500-5000 steps
            // - Threshold: 0.0001 (10% of early threshold)
            const pruneEvery = 1000; // Prune every 1000 steps
            shouldPrune = (updateStep % pruneEvery === 0);
            pruneThreshold = 0.0001;
        }

        // Execute pruning if scheduled
        if (shouldPrune) {
            const activeIndices = [];
            for (let i = 0; i < spectrum.amplitudes.length; i++) {
                if (spectrum.amplitudes[i] > pruneThreshold) {
                    activeIndices.push(i);
                }
            }

            spectrum.frequencies = activeIndices.map(i => spectrum.frequencies[i]);
            spectrum.amplitudes = activeIndices.map(i => spectrum.amplitudes[i]);
        }
    }

    /**
     * Measure compatibility between word spectrum and context measurement pattern
     * This is the core "measurement" operation
     *
     * Now simplified to real-valued dot product
     */
    measure(wordId, contextPattern) {
        const wordVector = this.toDenseVector(wordId);

        // Simple dot product (real-valued)
        let score = 0;

        for (let i = 0; i < this.frequencyDim; i++) {
            score += wordVector[i] * contextPattern[i];
        }

        return score;
    }

    /**
     * Normalize amplitudes to prevent runaway spectra
     *
     * Because amplitudes grow/shrink unpredictably during training,
     * normalization keeps pruning thresholds meaningful and prevents
     * numerical instability.
     *
     * @param {number} wordId
     * @param {string} method - 'sum' (L1) or 'euclidean' (L2)
     */
    normalizeSpectrum(wordId, method = 'euclidean') {
        const spectrum = this.getSpectrum(wordId);

        if (spectrum.frequencies.length === 0) {
            return; // Nothing to normalize
        }

        let norm = 0;

        if (method === 'sum') {
            // L1 normalization: sum of amplitudes = 1
            norm = spectrum.amplitudes.reduce((sum, amp) => sum + amp, 0);
        } else {
            // L2 normalization (Euclidean): sum of squares = 1
            norm = Math.sqrt(spectrum.amplitudes.reduce((sum, amp) => sum + amp * amp, 0));
        }

        if (norm > 1e-10) {
            for (let i = 0; i < spectrum.amplitudes.length; i++) {
                spectrum.amplitudes[i] /= norm;
            }
        }
    }

    /**
     * Get number of active frequencies for a word (sparsity measure)
     */
    getSparsity(wordId) {
        const spectrum = this.getSpectrum(wordId);
        return spectrum.frequencies.length;
    }

    /**
     * Get the dominant frequencies for a word (for interpretation)
     */
    getDominantFrequencies(wordId, topK = 3) {
        const spectrum = this.getSpectrum(wordId);

        const freqAmpPairs = spectrum.frequencies.map((freq, i) => ({
            frequency: freq,
            amplitude: spectrum.amplitudes[i]
        }));

        // Sort by amplitude
        freqAmpPairs.sort((a, b) => b.amplitude - a.amplitude);

        return freqAmpPairs.slice(0, topK);
    }
}
