/**
 * Stability Analysis Script
 *
 * Usage:
 *   node src/analyze_stability.js checkpoint1.json checkpoint2.json
 *
 * Analyzes spectrum stability between two training checkpoints
 */

import { StabilityAnalyzer } from './analysis/StabilityAnalyzer.js';
import { ModelPersistence } from './utils/ModelPersistence.js';
import fs from 'fs';
import path from 'path';

// Parse command line arguments
const args = process.argv.slice(2);

if (args.length < 2) {
  console.log('\n' + '█'.repeat(70));
  console.log('STABILITY ANALYSIS TOOL');
  console.log('█'.repeat(70) + '\n');

  console.log('Usage:');
  console.log('  node src/analyze_stability.js checkpoint1.json checkpoint2.json [word1,word2,...]');
  console.log('\nExamples:');
  console.log('  # Compare two checkpoints (auto-selects frequent words)');
  console.log('  node src/analyze_stability.js model_step10000.json model_step20000.json');
  console.log('\n  # Analyze specific words');
  console.log('  node src/analyze_stability.js model_step10000.json model_step20000.json bank,river,money');
  console.log('\n  # Analyze multiple checkpoints (time series)');
  console.log('  node src/analyze_stability.js model_step*.json\n');

  process.exit(1);
}

async function analyzeStability() {
  const checkpoint1Path = args[0];
  const checkpoint2Path = args[1];
  const probeWords = args[2] ? args[2].split(',') : null;

  // Validate files exist
  if (!fs.existsSync(checkpoint1Path)) {
    console.error(`❌ Checkpoint 1 not found: ${checkpoint1Path}`);
    process.exit(1);
  }

  if (!fs.existsSync(checkpoint2Path)) {
    console.error(`❌ Checkpoint 2 not found: ${checkpoint2Path}`);
    process.exit(1);
  }

  console.log('\n' + '█'.repeat(70));
  console.log('STABILITY ANALYSIS TOOL');
  console.log('█'.repeat(70) + '\n');

  console.log('Comparing checkpoints:');
  console.log(`  Checkpoint 1: ${path.basename(checkpoint1Path)}`);
  console.log(`  Checkpoint 2: ${path.basename(checkpoint2Path)}`);

  if (probeWords) {
    console.log(`  Probe words: ${probeWords.join(', ')}`);
  } else {
    console.log(`  Probe words: auto-select top 20 frequent words`);
  }
  console.log('');

  // Run analysis
  const result = StabilityAnalyzer.compareCheckpoints(
    checkpoint1Path,
    checkpoint2Path,
    probeWords
  );

  // Print per-word details if requested
  if (probeWords && probeWords.length <= 10) {
    console.log('\n' + '='.repeat(70));
    console.log('PER-WORD DETAILS');
    console.log('='.repeat(70) + '\n');

    for (const comparison of result.results) {
      StabilityAnalyzer.printWordComparison(comparison);
    }
  }

  console.log('\n' + '█'.repeat(70));
  console.log('ANALYSIS COMPLETE');
  console.log('█'.repeat(70) + '\n');

  console.log('Next steps:');
  console.log('- If FROZEN: Increase learning rate or reduce sparsity penalty');
  console.log('- If UNSTABLE: Reduce learning rate or add momentum');
  console.log('- If LEARNING: Continue training and check stability later');
  console.log('- If STABILIZING: Model approaching convergence\n');
}

async function analyzeTimeSeries() {
  console.log('\n' + '█'.repeat(70));
  console.log('TIME SERIES STABILITY ANALYSIS');
  console.log('█'.repeat(70) + '\n');

  // Parse checkpoint paths from glob pattern
  const pattern = args[0];

  // Simple glob matching for model_step*.json
  const dir = path.dirname(pattern) || '.';
  const files = fs.readdirSync(dir)
    .filter(f => f.match(/model_step\d+\.json$/))
    .sort((a, b) => {
      const stepA = parseInt(a.match(/\d+/)[0]);
      const stepB = parseInt(b.match(/\d+/)[0]);
      return stepA - stepB;
    })
    .map(f => path.join(dir, f));

  if (files.length < 2) {
    console.error('❌ Need at least 2 checkpoint files matching pattern');
    process.exit(1);
  }

  console.log(`Found ${files.length} checkpoints:\n`);
  files.forEach((f, i) => {
    console.log(`  ${i + 1}. ${path.basename(f)}`);
  });
  console.log('');

  const probeWords = args[1] ? args[1].split(',') : null;

  // Run time series analysis
  const results = StabilityAnalyzer.analyzeStabilityTimeSeries(files, probeWords);

  console.log('\n' + '█'.repeat(70));
  console.log('TIME SERIES ANALYSIS COMPLETE');
  console.log('█'.repeat(70) + '\n');
}

// Check if time series mode (glob pattern)
if (args[0].includes('*')) {
  analyzeTimeSeries().catch(console.error);
} else {
  analyzeStability().catch(console.error);
}
