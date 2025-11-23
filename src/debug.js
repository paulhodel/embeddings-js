/**
 * Debug Script - Quick qualitative model inspection
 *
 * Usage:
 *   node src/debug.js <model_path> [command] [args...]
 *
 * Commands:
 *   word <word> [topK]              - Show spectrum for a word
 *   freq <freqIndex> [topK]         - Show top words using a frequency
 *   compare <word1,word2,...>       - Compare word spectra side by side
 *   freqs <freq1,freq2,...>         - Compare frequencies side by side
 *   interpretable [topN] [topWords] - Show most interpretable frequencies
 *   sparsity                        - Show sparsity statistics
 *   extreme [topK]                  - Show words with extreme sparsity
 *   evolve <word> <checkpoint1,checkpoint2,...> - Track word evolution
 *
 * Examples:
 *   node src/debug.js models/css_model.json word bank
 *   node src/debug.js models/css_model.json freq 37
 *   node src/debug.js models/css_model.json compare bank,river,money
 *   node src/debug.js models/css_model.json interpretable 10
 *   node src/debug.js models/css_model.json sparsity
 */

import { DebugUtils } from './utils/DebugUtils.js';
import { ModelPersistence } from './utils/ModelPersistence.js';
import { CSSTrainer } from './core/CSSTrainer.js';
import { Tokenizer } from './preprocessing/tokenizer.js';
import fs from 'fs';

// Parse command line arguments
const args = process.argv.slice(2);

if (args.length < 1) {
  printUsage();
  process.exit(1);
}

const modelPath = args[0];
const command = args[1] || 'help';

function printUsage() {
  console.log('\n' + '█'.repeat(70));
  console.log('DEBUG UTILITIES - Quick Model Inspection');
  console.log('█'.repeat(70) + '\n');

  console.log('Usage:');
  console.log('  node src/debug.js <model_path> [command] [args...]\n');

  console.log('Commands:');
  console.log('  word <word> [topK]              - Show spectrum for a word');
  console.log('  freq <freqIndex> [topK]         - Show top words using a frequency');
  console.log('  compare <word1,word2,...>       - Compare word spectra side by side');
  console.log('  freqs <freq1,freq2,...>         - Compare frequencies side by side');
  console.log('  interpretable [topN] [topWords] - Show most interpretable frequencies');
  console.log('  sparsity                        - Show sparsity statistics');
  console.log('  extreme [topK]                  - Show words with extreme sparsity');
  console.log('  evolve <word> <path1,path2,...> - Track word spectrum evolution\n');

  console.log('Examples:');
  console.log('  # Show spectrum for "bank"');
  console.log('  node src/debug.js models/css_model.json word bank\n');

  console.log('  # Show top words for frequency 37');
  console.log('  node src/debug.js models/css_model.json freq 37\n');

  console.log('  # Compare word spectra');
  console.log('  node src/debug.js models/css_model.json compare bank,river,money\n');

  console.log('  # Show most interpretable frequencies');
  console.log('  node src/debug.js models/css_model.json interpretable 10\n');

  console.log('  # Show sparsity statistics');
  console.log('  node src/debug.js models/css_model.json sparsity\n');

  console.log('  # Track word evolution across checkpoints');
  console.log('  node src/debug.js models/snapshots/model_epoch2.json evolve bank models/snapshots/model_epoch2.json,models/snapshots/model_epoch4.json,models/snapshots/model_epoch6.json\n');
}

async function runDebugCommand() {
  if (command === 'help' || command === '--help' || command === '-h') {
    printUsage();
    return;
  }

  // Validate model path
  if (!fs.existsSync(modelPath)) {
    console.error(`❌ Model file not found: ${modelPath}`);
    process.exit(1);
  }

  // Load model
  console.log(`Loading model from: ${modelPath}...\n`);
  const model = ModelPersistence.loadModel(modelPath);

  // Reconstruct trainer
  const trainer = new CSSTrainer(model.config);
  trainer.initialize(model.modelData.vocabSize);
  trainer.importModel(model.modelData);

  // Reconstruct tokenizer
  const tokenizer = new Tokenizer();
  tokenizer.vocab = new Map(Object.entries(model.vocab));
  tokenizer.wordFreq = new Map(Object.entries(model.wordFreq));
  tokenizer.nextId = tokenizer.vocab.size;

  for (const [word, id] of tokenizer.vocab.entries()) {
    tokenizer.idToWord.set(id, word);
  }

  console.log(`✓ Model loaded: ${tokenizer.vocabSize} words, ${model.config.frequencyDim} frequencies\n`);

  // Execute command
  switch (command) {
    case 'word': {
      const word = args[2];
      if (!word) {
        console.error('❌ Usage: word <word> [topK]');
        process.exit(1);
      }
      const topK = args[3] ? parseInt(args[3]) : null;

      console.log('='.repeat(70));
      console.log('WORD SPECTRUM');
      console.log('='.repeat(70) + '\n');

      DebugUtils.printWordSpectrum(word, trainer, tokenizer, topK);
      console.log('');
      break;
    }

    case 'freq': {
      const freqIndex = parseInt(args[2]);
      if (isNaN(freqIndex)) {
        console.error('❌ Usage: freq <freqIndex> [topK]');
        process.exit(1);
      }
      const topK = args[3] ? parseInt(args[3]) : 10;

      console.log('='.repeat(70));
      console.log('FREQUENCY REVERSE LOOKUP');
      console.log('='.repeat(70));

      DebugUtils.showTopWordsForFrequency(freqIndex, trainer, tokenizer, topK);
      break;
    }

    case 'compare': {
      const wordsStr = args[2];
      if (!wordsStr) {
        console.error('❌ Usage: compare <word1,word2,...>');
        process.exit(1);
      }
      const words = wordsStr.split(',');
      const topK = args[3] ? parseInt(args[3]) : 5;

      DebugUtils.printWordSpectraComparison(words, trainer, tokenizer, topK);
      break;
    }

    case 'freqs': {
      const freqsStr = args[2];
      if (!freqsStr) {
        console.error('❌ Usage: freqs <freq1,freq2,...>');
        process.exit(1);
      }
      const frequencies = freqsStr.split(',').map(s => parseInt(s.trim()));
      const topK = args[3] ? parseInt(args[3]) : 8;

      DebugUtils.showFrequencyComparison(frequencies, trainer, tokenizer, topK);
      break;
    }

    case 'interpretable': {
      const topFreqs = args[2] ? parseInt(args[2]) : 5;
      const topWords = args[3] ? parseInt(args[3]) : 8;

      DebugUtils.showMostInterpretableFrequencies(trainer, tokenizer, topFreqs, topWords);
      break;
    }

    case 'sparsity': {
      DebugUtils.printSparsityStats(trainer, tokenizer);
      break;
    }

    case 'extreme': {
      const topK = args[2] ? parseInt(args[2]) : 10;
      DebugUtils.showExtremeSparsity(trainer, tokenizer, topK);
      break;
    }

    case 'evolve': {
      const word = args[2];
      const checkpointsStr = args[3];

      if (!word || !checkpointsStr) {
        console.error('❌ Usage: evolve <word> <checkpoint1,checkpoint2,...>');
        process.exit(1);
      }

      const checkpoints = checkpointsStr.split(',');
      const topK = args[4] ? parseInt(args[4]) : 5;

      DebugUtils.trackWordEvolution(word, checkpoints, topK);
      break;
    }

    default:
      console.error(`❌ Unknown command: ${command}`);
      console.log('\nRun with --help to see available commands');
      process.exit(1);
  }
}

// Run debug command
runDebugCommand().catch(error => {
  console.error('\n❌ Error:');
  console.error(error);
  process.exit(1);
});
