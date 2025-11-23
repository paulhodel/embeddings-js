/**
 * Vocabulary Preparation Script
 *
 * This script explores the entire corpus to build a complete vocabulary
 * and initializes all words with random sparse spectra before training begins.
 *
 * Benefits:
 * 1. No blank/uninitialized words in vocabulary
 * 2. Consistent initialization across all words
 * 3. Optional frequency-based or Gaussian initialization strategies
 * 4. Faster training (no lazy initialization overhead)
 *
 * Usage:
 *   node src/prepare_vocabulary.js [--strategy uniform|gaussian|frequency-scaled]
 */

import fs from 'fs';
import path from 'path';
import {execSync} from 'child_process';
import * as vocabulary from './vocabulary.js';

// ============================================
// CONFIGURATION
// ============================================

const CONFIG = {
    parquetDir: './data/parquet',
    vocabularyFile: './data/vocabulary.json',

    // CSS Model parameters (should match train.js)
    frequencyDim: 256,
    minInitFrequencies: 4,      // Variable initialization: start sparse
    maxInitFrequencies: 8,

    // Initialization strategies
    initStrategy: 'gaussian',  // 'uniform', 'gaussian', 'frequency-scaled'

    // Uniform initialization [min, max]
    uniformAmpMin: 0.0005,
    uniformAmpMax: 0.003,

    // Gaussian initialization (mean, stddev)
    gaussianMean: 0.001,
    gaussianStdDev: 0.0005,

    // Frequency-scaled: popular words get smaller initial amplitudes
    // (prevents them from dominating early training)
    frequencyScaleEnabled: true,
    frequencyScalePower: 0.5,  // sqrt scaling by default

    // Minimum word frequency to include in vocabulary
    minFrequency: 2,

    // Logging
    logEvery: 1000,
};

// ============================================
// INITIALIZATION STRATEGIES
// ============================================

/**
 * Generate random amplitude using uniform distribution
 */
function uniformAmplitude() {
    return Math.random() * (CONFIG.uniformAmpMax - CONFIG.uniformAmpMin) + CONFIG.uniformAmpMin;
}

/**
 * Generate random amplitude using Gaussian (Box-Muller transform)
 */
function gaussianAmplitude() {
    // Box-Muller transform for Gaussian random numbers
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);

    // Scale and shift to desired mean and stddev
    let amp = z0 * CONFIG.gaussianStdDev + CONFIG.gaussianMean;

    // Ensure positive amplitude
    amp = Math.max(0.0001, amp);

    return amp;
}

/**
 * Scale amplitude based on word frequency
 * More frequent words get smaller initial amplitudes to prevent early dominance
 */
function scaleAmplitudeByFrequency(baseAmplitude, frequency, totalWords) {
    if (!CONFIG.frequencyScaleEnabled) return baseAmplitude;

    // Normalized frequency [0, 1]
    const normalizedFreq = frequency / totalWords;

    // Scale factor: frequent words get smaller amplitudes
    // Using power < 1 (e.g., 0.5) means sqrt scaling
    const scaleFactor = Math.pow(1.0 - normalizedFreq, CONFIG.frequencyScalePower);

    return baseAmplitude * (0.5 + 0.5 * scaleFactor);  // Scale between 0.5x and 1.0x
}

/**
 * Initialize a word with random sparse spectrum (variable size 4-8)
 */
function initializeSpectrum(wordFrequency, totalWords) {
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

            // Generate base amplitude according to strategy
            let amp;
            switch (CONFIG.initStrategy) {
                case 'uniform':
                    amp = uniformAmplitude();
                    break;
                case 'gaussian':
                    amp = gaussianAmplitude();
                    break;
                case 'frequency-scaled':
                    amp = gaussianAmplitude();
                    amp = scaleAmplitudeByFrequency(amp, wordFrequency, totalWords);
                    break;
                default:
                    amp = uniformAmplitude();
            }

            amplitudes.push(amp);

            // Random phase [0, 2π]
            phases.push(Math.random() * 2 * Math.PI);
        }
    }

    return { frequencies, amplitudes, phases };
}

// ============================================
// CORPUS EXPLORATION
// ============================================

/**
 * Load parquet file
 */
function loadParquetFile(filepath) {
    console.log(`Loading: ${filepath}`);

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

/**
 * Get all parquet files
 */
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

/**
 * Tokenize and count words in corpus
 */
function buildVocabularyFromCorpus() {
    console.log('\n' + '='.repeat(70));
    console.log('BUILDING VOCABULARY FROM CORPUS');
    console.log('='.repeat(70) + '\n');

    const wordCounts = new Map();
    let totalTokens = 0;
    let documentCount = 0;

    const parquetFiles = getParquetFiles();
    if (parquetFiles.length === 0) {
        console.error('No parquet files found!');
        process.exit(1);
    }

    console.log(`Found ${parquetFiles.length} parquet file(s)\n`);

    for (const filepath of parquetFiles) {
        const documents = loadParquetFile(filepath);

        for (const content of documents) {
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
                        wordCounts.set(token, (wordCounts.get(token) || 0) + 1);
                        totalTokens++;
                        token = '';
                    }
                }
            }

            if (token.length > 0) {
                wordCounts.set(token, (wordCounts.get(token) || 0) + 1);
                totalTokens++;
            }

            documentCount++;
            if (documentCount % CONFIG.logEvery === 0) {
                console.log(`  Processed ${documentCount} documents, ${wordCounts.size} unique words, ${totalTokens} tokens`);
            }
        }
    }

    console.log(`\nCorpus statistics:`);
    console.log(`  Total documents: ${documentCount}`);
    console.log(`  Total tokens: ${totalTokens}`);
    console.log(`  Unique words: ${wordCounts.size}`);

    // Filter by minimum frequency
    const filteredWords = new Map();
    for (const [word, count] of wordCounts.entries()) {
        if (count >= CONFIG.minFrequency) {
            filteredWords.set(word, count);
        }
    }

    console.log(`  Words after min frequency filter (>=${CONFIG.minFrequency}): ${filteredWords.size}\n`);

    return { wordCounts: filteredWords, totalTokens };
}

/**
 * Initialize vocabulary with spectra
 */
function initializeVocabulary(wordCounts, totalTokens) {
    console.log('='.repeat(70));
    console.log('INITIALIZING WORD SPECTRA');
    console.log('='.repeat(70) + '\n');

    console.log(`Initialization strategy: ${CONFIG.initStrategy}`);
    console.log(`Frequency dimension: ${CONFIG.frequencyDim}`);
    console.log(`Max frequencies per word: ${CONFIG.maxFrequencies}\n`);

    vocabulary.clearVocabulary();

    let wordIndex = 0;
    const sortedWords = Array.from(wordCounts.entries()).sort((a, b) => b[1] - a[1]);

    for (const [word, count] of sortedWords) {
        // Add word to vocabulary
        const wordObj = vocabulary.addWord(word);

        // Initialize spectrum
        wordObj.spectrum = initializeSpectrum(count, totalTokens);
        wordObj.frequency = count;  // Store frequency for analysis

        wordIndex++;
        if (wordIndex % 1000 === 0) {
            console.log(`  Initialized ${wordIndex}/${sortedWords.length} words`);
        }
    }

    console.log(`\n✓ Initialized ${vocabulary.getVocabSize()} words\n`);

    // Show statistics
    console.log('Initialization statistics:');

    const allWords = vocabulary.getAllWords();
    const allAmplitudes = [];
    for (const wordObj of allWords) {
        if (wordObj.spectrum && wordObj.spectrum.amplitudes) {
            allAmplitudes.push(...wordObj.spectrum.amplitudes);
        }
    }

    if (allAmplitudes.length > 0) {
        allAmplitudes.sort((a, b) => a - b);
        const mean = allAmplitudes.reduce((sum, a) => sum + a, 0) / allAmplitudes.length;
        const variance = allAmplitudes.reduce((sum, a) => sum + Math.pow(a - mean, 2), 0) / allAmplitudes.length;
        const stddev = Math.sqrt(variance);
        const median = allAmplitudes[Math.floor(allAmplitudes.length / 2)];

        console.log(`  Amplitude distribution:`);
        console.log(`    Mean: ${mean.toFixed(6)}`);
        console.log(`    StdDev: ${stddev.toFixed(6)}`);
        console.log(`    Median: ${median.toFixed(6)}`);
        console.log(`    Min: ${allAmplitudes[0].toFixed(6)}`);
        console.log(`    Max: ${allAmplitudes[allAmplitudes.length - 1].toFixed(6)}`);
    }

    // Show top 10 most frequent words
    console.log(`\n  Top 10 most frequent words:`);
    sortedWords.slice(0, 10).forEach(([word, count], idx) => {
        const wordObj = vocabulary.getWordByString(word);
        const avgAmp = wordObj.spectrum.amplitudes.reduce((sum, a) => sum + a, 0) / wordObj.spectrum.amplitudes.length;
        console.log(`    ${idx + 1}. "${word}": ${count} occurrences, avg amplitude: ${avgAmp.toFixed(6)}`);
    });
}

// ============================================
// MAIN
// ============================================

async function main() {
    console.log('\n' + '█'.repeat(70));
    console.log('VOCABULARY PREPARATION');
    console.log('█'.repeat(70) + '\n');

    // Parse command line arguments
    const args = process.argv.slice(2);
    for (const arg of args) {
        if (arg.startsWith('--strategy=')) {
            CONFIG.initStrategy = arg.split('=')[1];
        } else if (arg === '--strategy') {
            const nextArg = args[args.indexOf(arg) + 1];
            if (nextArg) CONFIG.initStrategy = nextArg;
        }
    }

    console.log('Configuration:');
    console.log(`  Parquet directory: ${CONFIG.parquetDir}`);
    console.log(`  Output file: ${CONFIG.vocabularyFile}`);
    console.log(`  Initialization strategy: ${CONFIG.initStrategy}`);
    console.log(`  Min word frequency: ${CONFIG.minFrequency}`);
    console.log('');

    // Step 1: Explore corpus and build vocabulary
    const { wordCounts, totalTokens } = buildVocabularyFromCorpus();

    // Step 2: Initialize all words with random spectra
    initializeVocabulary(wordCounts, totalTokens);

    // Step 3: Save vocabulary
    console.log('='.repeat(70));
    console.log('SAVING VOCABULARY');
    console.log('='.repeat(70) + '\n');

    vocabulary.saveVocabulary();

    console.log('\n' + '█'.repeat(70));
    console.log('VOCABULARY PREPARATION COMPLETE');
    console.log('█'.repeat(70) + '\n');

    console.log('Next steps:');
    console.log('  1. Run training: node src/train.js');
    console.log('  2. Training will use the pre-built vocabulary');
    console.log('  3. All words are already initialized with random spectra\n');
}

// Run main
main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});
