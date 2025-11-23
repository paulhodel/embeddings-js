/**
 * DebugUtils - Quick qualitative debugging functions
 *
 * Provides fast visualization tools for inspecting learned spectra:
 * - printWordSpectrum: Show a word's active frequencies and amplitudes
 * - showTopWordsForFrequency: Reverse lookup - which words use a frequency
 */

export class DebugUtils {
    /**
     * Print a word's spectrum in human-readable format
     *
     * @param {string} word - Word to inspect
     * @param {CSSTrainer} trainer - Trained CSS model
     * @param {Tokenizer} tokenizer - Tokenizer with vocabulary
     * @param {number} topK - Number of frequencies to show (default: all)
     */
    static printWordSpectrum(word, trainer, tokenizer, topK = null) {
        const wordId = tokenizer.wordToId(word);

        if (wordId === undefined) {
            console.log(`❌ Word "${word}" not in vocabulary`);
            return;
        }

        const spectrum = trainer.spectralWord.getSpectrum(wordId);

        if (spectrum.frequencies.length === 0) {
            console.log(`"${word}": (no active frequencies)`);
            return;
        }

        // Sort by amplitude (descending)
        const sorted = spectrum.frequencies
            .map((freq, idx) => ({
                freq,
                amp: spectrum.amplitudes[idx]
            }))
            .sort((a, b) => b.amp - a.amp);

        // Limit to top K if specified
        const toShow = topK ? sorted.slice(0, topK) : sorted;

        console.log(`"${word}":`);
        for (const {freq, amp} of toShow) {
            console.log(`  (freq ${freq.toString().padStart(4)}) amp=${amp.toFixed(4)}`);
        }
    }

    /**
     * Print multiple word spectra side by side for comparison
     *
     * @param {Array<string>} words - Words to compare
     * @param {CSSTrainer} trainer - Trained CSS model
     * @param {Tokenizer} tokenizer - Tokenizer with vocabulary
     * @param {number} topK - Number of frequencies to show per word
     */
    static printWordSpectraComparison(words, trainer, tokenizer, topK = 5) {
        console.log('\n' + '='.repeat(70));
        console.log('WORD SPECTRA COMPARISON');
        console.log('='.repeat(70) + '\n');

        for (const word of words) {
            this.printWordSpectrum(word, trainer, tokenizer, topK);
            console.log('');
        }
    }

    /**
     * Show top words for a given frequency (reverse lookup)
     *
     * @param {number} freqIndex - Frequency to inspect
     * @param {CSSTrainer} trainer - Trained CSS model
     * @param {Tokenizer} tokenizer - Tokenizer with vocabulary
     * @param {number} topK - Number of top words to show (default: 10)
     * @param {number} minAmp - Minimum amplitude threshold (default: 0.01)
     */
    static showTopWordsForFrequency(freqIndex, trainer, tokenizer, topK = 10, minAmp = 0.01) {
        console.log(`\nFrequency ${freqIndex}:`);

        // Collect all words using this frequency
        const wordsWithFreq = [];

        for (const [word, wordId] of tokenizer.vocab.entries()) {
            const spectrum = trainer.spectralWord.getSpectrum(wordId);
            const idx = spectrum.frequencies.indexOf(freqIndex);

            if (idx !== -1) {
                const amp = spectrum.amplitudes[idx];
                if (amp >= minAmp) {
                    wordsWithFreq.push({word, amp});
                }
            }
        }

        // Sort by amplitude (descending)
        wordsWithFreq.sort((a, b) => b.amp - a.amp);

        // Limit to top K
        const topWords = wordsWithFreq.slice(0, topK);

        if (topWords.length === 0) {
            console.log(`  (no words with amplitude >= ${minAmp})`);
            return;
        }

        for (const {word, amp} of topWords) {
            console.log(`  ${word.padEnd(20)} (${amp.toFixed(4)})`);
        }

        console.log(`  ... ${wordsWithFreq.length} total words use this frequency\n`);
    }

    /**
     * Show multiple frequencies side by side
     *
     * @param {Array<number>} frequencies - Frequencies to inspect
     * @param {CSSTrainer} trainer - Trained CSS model
     * @param {Tokenizer} tokenizer - Tokenizer with vocabulary
     * @param {number} topK - Number of top words per frequency
     */
    static showFrequencyComparison(frequencies, trainer, tokenizer, topK = 8) {
        console.log('\n' + '='.repeat(70));
        console.log('FREQUENCY COMPARISON - Top Words per Frequency');
        console.log('='.repeat(70));

        for (const freq of frequencies) {
            this.showTopWordsForFrequency(freq, trainer, tokenizer, topK);
        }
    }

    /**
     * Find the most interpretable frequencies (highest variance in word usage)
     *
     * @param {CSSTrainer} trainer - Trained CSS model
     * @param {Tokenizer} tokenizer - Tokenizer with vocabulary
     * @param {number} topK - Number of frequencies to return
     * @returns {Array<{freq: number, numWords: number, variance: number}>}
     */
    static findMostInterpretableFrequencies(trainer, tokenizer, topK = 10) {
        const freqStats = new Map(); // freq -> {amps: [], numWords}

        // Collect amplitude statistics for each frequency
        for (const [word, wordId] of tokenizer.vocab.entries()) {
            const spectrum = trainer.spectralWord.getSpectrum(wordId);

            for (let i = 0; i < spectrum.frequencies.length; i++) {
                const freq = spectrum.frequencies[i];
                const amp = spectrum.amplitudes[i];

                if (!freqStats.has(freq)) {
                    freqStats.set(freq, {amps: [], numWords: 0});
                }

                const stats = freqStats.get(freq);
                stats.amps.push(amp);
                stats.numWords++;
            }
        }

        // Compute variance for each frequency
        const freqVariances = [];

        for (const [freq, stats] of freqStats.entries()) {
            if (stats.numWords < 3) continue; // Skip rare frequencies

            const mean = stats.amps.reduce((a, b) => a + b, 0) / stats.amps.length;
            const variance = stats.amps.reduce((sum, amp) => sum + (amp - mean) ** 2, 0) / stats.amps.length;

            freqVariances.push({
                freq,
                numWords: stats.numWords,
                variance,
                maxAmp: Math.max(...stats.amps)
            });
        }

        // Sort by variance (descending) - high variance = more interpretable
        freqVariances.sort((a, b) => b.variance - a.variance);

        return freqVariances.slice(0, topK);
    }

    /**
     * Show the most interpretable frequencies and their top words
     *
     * @param {CSSTrainer} trainer - Trained CSS model
     * @param {Tokenizer} tokenizer - Tokenizer with vocabulary
     * @param {number} topFreqs - Number of frequencies to show
     * @param {number} topWords - Number of words per frequency
     */
    static showMostInterpretableFrequencies(trainer, tokenizer, topFreqs = 5, topWords = 8) {
        console.log('\n' + '█'.repeat(70));
        console.log('MOST INTERPRETABLE FREQUENCIES');
        console.log('█'.repeat(70));
        console.log('\nFrequencies with highest variance = clearest semantic factors\n');

        const interpretable = this.findMostInterpretableFrequencies(trainer, tokenizer, topFreqs);

        for (let i = 0; i < interpretable.length; i++) {
            const {freq, numWords, variance, maxAmp} = interpretable[i];

            console.log(`${i + 1}. Frequency ${freq}`);
            console.log(`   Words using it: ${numWords}`);
            console.log(`   Variance: ${variance.toFixed(4)} (higher = more interpretable)`);
            console.log(`   Max amplitude: ${maxAmp.toFixed(4)}\n`);

            console.log(`   Top words:`);
            this.showTopWordsForFrequency(freq, trainer, tokenizer, topWords, 0.05);
        }
    }

    /**
     * Track spectrum evolution for a word across multiple checkpoints
     *
     * @param {string} word - Word to track
     * @param {Array<string>} checkpointPaths - Paths to checkpoint files
     * @param {number} topK - Number of frequencies to show per checkpoint
     */
    static trackWordEvolution(word, checkpointPaths, topK = 5) {
        console.log('\n' + '='.repeat(70));
        console.log(`TRACKING SPECTRUM EVOLUTION: "${word}"`);
        console.log('='.repeat(70) + '\n');

        const ModelPersistence = require('./ModelPersistence.js').ModelPersistence;
        const CSSTrainer = require('../core/CSSTrainer.js').CSSTrainer;

        for (let i = 0; i < checkpointPaths.length; i++) {
            const path = checkpointPaths[i];
            const filename = path.split('/').pop();

            console.log(`Checkpoint ${i + 1}: ${filename}`);

            try {
                const model = ModelPersistence.loadModel(path);
                const trainer = new CSSTrainer(model.config);
                trainer.initialize(model.modelData.vocabSize);
                trainer.importModel(model.modelData);

                const wordId = model.vocab.get(word);
                if (wordId === undefined) {
                    console.log(`  ⚠️  Word not in vocabulary\n`);
                    continue;
                }

                const spectrum = trainer.spectralWord.getSpectrum(wordId);

                if (spectrum.frequencies.length === 0) {
                    console.log(`  (no active frequencies)\n`);
                    continue;
                }

                // Sort by amplitude
                const sorted = spectrum.frequencies
                    .map((freq, idx) => ({freq, amp: spectrum.amplitudes[idx]}))
                    .sort((a, b) => b.amp - a.amp)
                    .slice(0, topK);

                for (const {freq, amp} of sorted) {
                    console.log(`    (freq ${freq.toString().padStart(4)}) amp=${amp.toFixed(4)}`);
                }

                console.log('');
            } catch (error) {
                console.log(`  ❌ Error loading checkpoint: ${error.message}\n`);
            }
        }

        console.log('='.repeat(70));
        console.log('Look for:');
        console.log('  ✓ New meaningful stable peaks appearing');
        console.log('  ✓ Noise peaks disappearing');
        console.log('  ✓ Spectrum shape stops jittering');
        console.log('='.repeat(70) + '\n');
    }

    /**
     * Print a summary of model sparsity statistics
     *
     * @param {CSSTrainer} trainer - Trained CSS model
     * @param {Tokenizer} tokenizer - Tokenizer with vocabulary
     */
    static printSparsityStats(trainer, tokenizer) {
        console.log('\n' + '='.repeat(70));
        console.log('SPARSITY STATISTICS');
        console.log('='.repeat(70) + '\n');

        const sparsities = [];
        let totalActiveFreqs = 0;

        for (const [word, wordId] of tokenizer.vocab.entries()) {
            const spectrum = trainer.spectralWord.getSpectrum(wordId);
            const numFreqs = spectrum.frequencies.length;
            sparsities.push(numFreqs);
            totalActiveFreqs += numFreqs;
        }

        sparsities.sort((a, b) => a - b);

        const mean = sparsities.reduce((a, b) => a + b, 0) / sparsities.length;
        const median = sparsities[Math.floor(sparsities.length / 2)];
        const min = sparsities[0];
        const max = sparsities[sparsities.length - 1];

        console.log(`Vocabulary size: ${tokenizer.vocabSize}`);
        console.log(`Total active frequencies: ${totalActiveFreqs}`);
        console.log(`\nActive frequencies per word:`);
        console.log(`  Mean:   ${mean.toFixed(2)}`);
        console.log(`  Median: ${median}`);
        console.log(`  Min:    ${min}`);
        console.log(`  Max:    ${max}`);

        // Histogram
        console.log(`\nDistribution:`);
        const histogram = {};
        for (const s of sparsities) {
            histogram[s] = (histogram[s] || 0) + 1;
        }

        const sortedBins = Object.keys(histogram).map(Number).sort((a, b) => a - b);
        for (const bin of sortedBins) {
            const count = histogram[bin];
            const pct = (count / sparsities.length * 100).toFixed(1);
            const bar = '█'.repeat(Math.floor(count / sparsities.length * 50));
            console.log(`  ${bin.toString().padStart(2)} freqs: ${count.toString().padStart(5)} words (${pct.padStart(5)}%) ${bar}`);
        }

        console.log('\n' + '='.repeat(70) + '\n');
    }

    /**
     * Find words with the most/least active frequencies
     *
     * @param {CSSTrainer} trainer - Trained CSS model
     * @param {Tokenizer} tokenizer - Tokenizer with vocabulary
     * @param {number} topK - Number of words to show
     */
    static showExtremeSparsity(trainer, tokenizer, topK = 10) {
        console.log('\n' + '='.repeat(70));
        console.log('EXTREME SPARSITY WORDS');
        console.log('='.repeat(70) + '\n');

        const wordSparsities = [];

        for (const [word, wordId] of tokenizer.vocab.entries()) {
            const spectrum = trainer.spectralWord.getSpectrum(wordId);
            wordSparsities.push({
                word,
                numFreqs: spectrum.frequencies.length
            });
        }

        // Sort by sparsity
        wordSparsities.sort((a, b) => b.numFreqs - a.numFreqs);

        console.log('MOST active frequencies (complex/polysemous words):');
        for (let i = 0; i < Math.min(topK, wordSparsities.length); i++) {
            const {word, numFreqs} = wordSparsities[i];
            console.log(`  ${(i + 1).toString().padStart(2)}. ${word.padEnd(20)} (${numFreqs} frequencies)`);
        }

        console.log('\nLEAST active frequencies (simple/monosemous words):');
        const reversed = [...wordSparsities].reverse();
        for (let i = 0; i < Math.min(topK, reversed.length); i++) {
            const {word, numFreqs} = reversed[i];
            if (numFreqs === 0) continue; // Skip uninitialized words
            console.log(`  ${(i + 1).toString().padStart(2)}. ${word.padEnd(20)} (${numFreqs} frequency)`);
        }

        console.log('\n' + '='.repeat(70) + '\n');
    }
}
