/**
 * Polysemy Analysis Script
 *
 * Usage:
 *   node src/analyze_polysemy.js
 *
 * Analyzes sense separation for polysemous words like "bank"
 */

import { Tokenizer } from './preprocessing/tokenizer.js';
import { CSSTrainer } from './core/CSSTrainer.js';
import { PolysemyAnalyzer } from './analysis/PolysemyAnalyzer.js';

// Sample corpus with "bank" in different senses
const sampleTexts = [
  // Financial sense
  "the bank lends money to businesses and individuals",
  "he deposits money in the bank account every month",
  "the bank offers loans with low interest rates",
  "she works at the financial bank downtown",
  "the bank provides credit cards and mortgages",
  "customers visit the bank to withdraw cash",
  "the bank's assets grew significantly this year",
  "online banking makes it easy to transfer money",
  "the central bank sets the interest rate policy",
  "the bank approved his loan application quickly",

  // River sense
  "the river bank is covered with green grass",
  "we sat on the bank and watched the water flow",
  "the bank of the river is eroding slowly",
  "fish swim near the bank where plants grow",
  "the boat is moored at the river bank",
  "children play on the grassy bank by the stream",
  "the bank along the river provides habitat for birds",
  "we walked along the muddy bank of the creek",
  "the bank collapsed after heavy rains",
  "trees line the bank of the lake",

  // Mixed contexts (some with both meanings in different sentences)
  "the cat sleeps on the mat near the window",
  "the dog runs in the park every morning",
  "people walk together through the city streets",
  "the sun shines brightly in the sky today",
  "flowers bloom in the garden during spring",
];

async function analyzePolysemy() {
  console.log('\n' + '█'.repeat(70));
  console.log('POLYSEMY ANALYSIS DEMONSTRATION');
  console.log('█'.repeat(70) + '\n');

  console.log('This script demonstrates sense separation analysis for polysemous words.');
  console.log('Example: "bank" (financial institution vs. river bank)\n');

  // Step 1: Build vocabulary and corpus
  console.log('Step 1: Preparing corpus...');
  const tokenizer = new Tokenizer();
  tokenizer.buildVocab(sampleTexts, 1);

  const corpus = sampleTexts
    .map(text => tokenizer.textToIds(text))
    .filter(ids => ids.length > 0);

  console.log(`  Vocabulary size: ${tokenizer.vocabSize}`);
  console.log(`  Corpus size: ${corpus.length} documents\n`);

  // Check if "bank" is in vocabulary
  const bankId = tokenizer.wordToId('bank');
  if (bankId === undefined) {
    console.log('❌ Word "bank" not found in vocabulary');
    return;
  }

  // Count "bank" occurrences
  let bankCount = 0;
  for (const doc of corpus) {
    bankCount += doc.filter(id => id === bankId).length;
  }
  console.log(`  "bank" appears ${bankCount} times in corpus\n`);

  // Step 2: Train CSS model
  console.log('Step 2: Training CSS model...');
  const trainer = new CSSTrainer({
    frequencyDim: 50,
    maxFrequencies: 5,
    windowSize: 3,
    learningRate: 0.05,
    sparsityPenalty: 0.002,
    epochs: 20,
    batchSize: 50,
    negativeCount: 5,
    margin: 0.5,
    updateNegatives: true
  });

  trainer.initialize(tokenizer.vocabSize);
  trainer.train(corpus);

  // Step 3: Analyze polysemy
  console.log('\n\nStep 3: Analyzing sense separation...\n');

  const analyzer = new PolysemyAnalyzer(trainer, tokenizer);

  // Analyze "bank"
  const result = analyzer.analyzeSenseSeparation('bank', corpus, 2);

  if (result) {
    // Additional analysis: Show context clusters in detail
    console.log('\n\n' + '='.repeat(70));
    console.log('DETAILED CONTEXT ANALYSIS');
    console.log('='.repeat(70) + '\n');

    for (let cluster = 0; cluster < 2; cluster++) {
      console.log(`Cluster ${cluster + 1} contexts:`);

      const clusterContexts = result.contexts.filter((_, idx) =>
        result.clusters.assignments[idx] === cluster
      );

      clusterContexts.slice(0, 5).forEach((ctx, idx) => {
        const contextWords = ctx.context
          .map(id => tokenizer.idToWord.get(id))
          .filter(w => w);

        console.log(`  ${idx + 1}. ${contextWords.join(' ')}`);
      });

      console.log('');
    }

    // Frequency overlap analysis
    console.log('\n' + '='.repeat(70));
    console.log('FREQUENCY OVERLAP ANALYSIS');
    console.log('='.repeat(70) + '\n');

    const wordFreqs = result.wordSpectrum.frequencies;
    console.log(`Word "bank" active frequencies: [${wordFreqs.join(', ')}]\n`);

    for (let i = 0; i < result.clusters.centroids.length; i++) {
      const centroid = result.clusters.centroids[i];
      const topFreqs = centroid
        .map((amp, freq) => ({ freq, amp }))
        .filter(p => p.amp > 0.01)
        .sort((a, b) => b.amp - a.amp)
        .slice(0, 5);

      console.log(`Cluster ${i + 1} top frequencies:`);
      topFreqs.forEach(f => {
        const isInWord = wordFreqs.includes(f.freq);
        const marker = isInWord ? '✓' : ' ';
        console.log(`  ${marker} Freq ${f.freq}: ${f.amp.toFixed(4)}`);
      });
      console.log('');
    }
  }

  console.log('\n' + '█'.repeat(70));
  console.log('ANALYSIS COMPLETE');
  console.log('█'.repeat(70) + '\n');

  console.log('Summary:');
  console.log('- Built context spectra for each "bank" occurrence');
  console.log('- Clustered contexts into 2 groups (financial vs. river)');
  console.log('- Checked if word frequencies align with cluster centroids');
  console.log('- Result indicates whether polysemy has emerged\n');
}

// Run analysis
analyzePolysemy().catch(console.error);
