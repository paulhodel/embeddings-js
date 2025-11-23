/**
 * Compressive Semantic Spectroscopy (CSS) - Main Entry Point
 *
 * A sparse spectral framework for learning word meaning
 * where words are represented as sparse spectra instead of dense vectors
 */

import {Tokenizer} from './preprocessing/tokenizer.js';
import {CSSTrainer} from './core/CSSTrainer.js';

// Sample corpus for demonstration
const sampleTexts = [
    "the king rules the kingdom with wisdom",
    "the queen governs the land with grace",
    "a man walks to the store",
    "a woman walks to the market",
    "the cat sleeps on the mat",
    "the dog sleeps on the floor",
    "kings and queens rule kingdoms",
    "men and women walk together",
    "cats and dogs are pets",
    "the river flows to the ocean",
    "water flows from the mountain",
    "the bank of the river is green",
    "money in the bank is safe",
    "the bank lends money to people",
    "people fish in the river",
    "the financial bank is downtown"
];

async function main() {
    console.log('\n========================================');
    console.log('COMPRESSIVE SEMANTIC SPECTROSCOPY');
    console.log('========================================\n');

    // Step 1: Tokenization and vocabulary building
    console.log('Step 1: Building vocabulary...');
    const tokenizer = new Tokenizer();
    tokenizer.buildVocab(sampleTexts, 1); // Min frequency = 1 for small corpus

    // Convert texts to word ID sequences
    const corpus = sampleTexts.map(text => tokenizer.textToIds(text));

    // Step 2: Initialize CSS trainer
    console.log('\nStep 2: Initializing CSS trainer...');
    const trainer = new CSSTrainer({
        frequencyDim: 50,        // Total frequency space dimension
        maxFrequencies: 5,       // Max active frequencies per word
        windowSize: 2,           // Context window size
        learningRate: 0.05,      // Learning rate
        sparsityPenalty: 0.002,  // L1 sparsity penalty
        epochs: 15,              // Training epochs
        batchSize: 50            // Batch size
    });

    trainer.initialize(tokenizer.vocabSize);

    // Step 3: Train the model
    console.log('\nStep 3: Training...');
    trainer.train(corpus);

    // Step 4: Analyze results
    console.log('\n========================================');
    console.log('ANALYSIS & INTERPRETATION');
    console.log('========================================\n');

    // Function to display word spectrum
    function displayWordSpectrum(word) {
        const wordId = tokenizer.wordToId(word);
        if (wordId === undefined) {
            console.log(`Word "${word}" not in vocabulary`);
            return;
        }

        const spectrum = trainer.getWordSpectrum(wordId, 5);
        const sparsity = trainer.spectralWord.getSparsity(wordId);

        console.log(`\nWord: "${word}" (ID: ${wordId})`);
        console.log(`  Active Frequencies: ${sparsity}`);
        console.log(`  Dominant Modes:`);

        spectrum.forEach((mode, idx) => {
            console.log(`    ${idx + 1}. Freq=${mode.frequency}, Amp=${mode.amplitude.toFixed(4)}`);
        });
    }

    // Function to find similar words
    function findSimilarWords(word, topK = 5) {
        const wordId = tokenizer.wordToId(word);
        if (wordId === undefined) {
            console.log(`Word "${word}" not in vocabulary`);
            return;
        }

        console.log(`\nMost similar to "${word}":`);
        const similar = trainer.findSimilar(wordId, topK);

        similar.forEach((item, idx) => {
            const similarWord = tokenizer.idToWord.get(item.wordId);
            console.log(`  ${idx + 1}. "${similarWord}" (similarity: ${item.similarity.toFixed(4)})`);
        });
    }

    // Analyze key words
    const wordsToAnalyze = ['bank', 'king', 'queen', 'river', 'cat', 'dog'];

    console.log('--- Word Spectra ---');
    wordsToAnalyze.forEach(word => {
        if (tokenizer.hasWord(word)) {
            displayWordSpectrum(word);
        }
    });

    console.log('\n\n--- Semantic Similarity ---');
    ['bank', 'king', 'river', 'cat'].forEach(word => {
        if (tokenizer.hasWord(word)) {
            findSimilarWords(word);
        }
    });

    // Demonstrate polysemy with "bank"
    console.log('\n\n--- Polysemy Analysis: "bank" ---');
    console.log('The word "bank" appears in both financial and river contexts.');
    console.log('CSS should learn multiple frequency peaks for different senses.\n');
    displayWordSpectrum('bank');

    // Export model
    console.log('\n\n--- Model Export ---');
    const model = trainer.exportModel();
    console.log(`Exported model with ${Object.keys(model.spectra).length} word spectra`);
    console.log(`Model size: ~${JSON.stringify(model).length} bytes`);

    console.log('\n========================================');
    console.log('DONE');
    console.log('========================================\n');
}

// Run the demo
main().catch(console.error);
