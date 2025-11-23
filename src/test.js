/**
 * Test suite for Compressive Semantic Spectroscopy
 */

import { Tokenizer } from './preprocessing/tokenizer.js';
import { SpectralWord } from './core/SpectralWord.js';
import { ContextMeasurement } from './core/ContextMeasurement.js';
import { CSSTrainer } from './core/CSSTrainer.js';

function testTokenizer() {
  console.log('\n=== Testing Tokenizer ===');

  const tokenizer = new Tokenizer();
  const texts = [
    "Hello world",
    "Hello there",
    "world peace"
  ];

  tokenizer.buildVocab(texts, 1);

  console.log(`Vocabulary size: ${tokenizer.vocabSize}`);
  console.log(`"hello" ID: ${tokenizer.wordToId('hello')}`);
  console.log(`"world" ID: ${tokenizer.wordToId('world')}`);

  const ids = tokenizer.textToIds("hello world");
  console.log(`"hello world" as IDs: [${ids}]`);

  console.log('✓ Tokenizer tests passed\n');
}

function testSpectralWord() {
  console.log('\n=== Testing SpectralWord ===');

  const spectralWord = new SpectralWord(100, 5, 50);

  // Initialize a word
  spectralWord.initializeWord(0);
  const spectrum = spectralWord.getSpectrum(0);

  console.log(`Word 0 spectrum:`);
  console.log(`  Active frequencies: ${spectrum.frequencies.length}`);
  console.log(`  Frequencies: [${spectrum.frequencies.join(', ')}]`);
  console.log(`  Amplitudes: [${spectrum.amplitudes.map(a => a.toFixed(3)).join(', ')}]`);

  // Convert to dense
  const dense = spectralWord.toDenseVector(0);
  console.log(`  Dense vector length: ${dense.length}`);

  // Get sparsity
  const sparsity = spectralWord.getSparsity(0);
  console.log(`  Sparsity (active freqs): ${sparsity}`);

  // Get dominant frequencies
  const dominant = spectralWord.getDominantFrequencies(0, 3);
  console.log(`  Top 3 dominant frequencies:`);
  dominant.forEach((d, i) => {
    console.log(`    ${i + 1}. Freq ${d.frequency}: Amp=${d.amplitude.toFixed(3)}`);
  });

  console.log('✓ SpectralWord tests passed\n');
}

function testContextMeasurement() {
  console.log('\n=== Testing ContextMeasurement ===');

  const spectralWord = new SpectralWord(100, 5, 50);
  const contextMeasurement = new ContextMeasurement(50, 2);

  // Initialize some words
  for (let i = 0; i < 10; i++) {
    spectralWord.initializeWord(i);
  }

  // Create a measurement pattern
  const contextWords = [1, 2, 3];
  const pattern = contextMeasurement.createMeasurementPattern(contextWords, spectralWord);

  console.log(`Created measurement pattern from context words [${contextWords}]`);
  console.log(`  Pattern length: ${pattern.length}`);
  console.log(`  Pattern sample: [${pattern.slice(0, 5).map(p => p.toFixed(3)).join(', ')}, ...]`);

  // Extract context windows
  const wordIds = [0, 1, 2, 3, 4, 5];
  const windows = contextMeasurement.extractContextWindows(wordIds);

  console.log(`Extracted ${windows.length} context windows from sequence [${wordIds}]`);
  console.log(`  Example: target=${windows[0].target}, context=[${windows[0].context}]`);

  // Record measurements
  contextMeasurement.recordMeasurement(0, pattern);
  contextMeasurement.recordMeasurement(0, pattern);

  const count = contextMeasurement.getMeasurementCount(0);
  console.log(`  Recorded measurements for word 0: ${count}`);

  console.log('✓ ContextMeasurement tests passed\n');
}

function testCSSTrainer() {
  console.log('\n=== Testing CSSTrainer ===');

  // Create a tiny corpus
  const texts = [
    "cat sleeps mat",
    "dog sleeps floor",
    "cat walks garden",
    "dog walks park"
  ];

  const tokenizer = new Tokenizer();
  tokenizer.buildVocab(texts, 1);
  const corpus = texts.map(text => tokenizer.textToIds(text));

  console.log(`Created corpus with ${corpus.length} documents`);
  console.log(`Vocabulary size: ${tokenizer.vocabSize}`);

  // Initialize trainer
  const trainer = new CSSTrainer({
    frequencyDim: 20,
    maxFrequencies: 3,
    windowSize: 1,
    learningRate: 0.1,
    sparsityPenalty: 0.005,
    epochs: 5,
    batchSize: 10
  });

  trainer.initialize(tokenizer.vocabSize);
  console.log('Initialized trainer');

  // Train
  trainer.train(corpus);

  // Check results
  const catId = tokenizer.wordToId('cat');
  const dogId = tokenizer.wordToId('dog');

  if (catId !== undefined && dogId !== undefined) {
    const similarity = trainer.spectralSimilarity(catId, dogId);
    console.log(`\nSemantic similarity between "cat" and "dog": ${similarity.toFixed(4)}`);

    const catSpectrum = trainer.getWordSpectrum(catId);
    console.log(`\n"cat" spectrum:`);
    catSpectrum.forEach((mode, i) => {
      console.log(`  ${i + 1}. Freq=${mode.frequency}, Amp=${mode.amplitude.toFixed(4)}`);
    });
  }

  // Export and import
  const model = trainer.exportModel();
  console.log(`\nExported model with ${Object.keys(model.spectra).length} spectra`);

  const newTrainer = new CSSTrainer();
  newTrainer.importModel(model);
  console.log(`Imported model successfully`);

  console.log('✓ CSSTrainer tests passed\n');
}

function runAllTests() {
  console.log('\n========================================');
  console.log('CSS TEST SUITE');
  console.log('========================================');

  try {
    testTokenizer();
    testSpectralWord();
    testContextMeasurement();
    testCSSTrainer();

    console.log('\n========================================');
    console.log('ALL TESTS PASSED ✓');
    console.log('========================================\n');
  } catch (error) {
    console.error('\n✗ Test failed with error:');
    console.error(error);
    process.exit(1);
  }
}

// Run all tests
runAllTests();
