/**
 * StabilityAnalyzer - Measure spectrum stability over training
 *
 * Compares word spectra between checkpoints to detect:
 * - Early: Big changes (still learning)
 * - Later: Smaller changes (stabilizing)
 * - Issues: Wild changes forever (unstable) or freeze too early (learning rate too low)
 */

import { ModelPersistence } from '../utils/ModelPersistence.js';
import { CSSTrainer } from '../core/CSSTrainer.js';
import { Tokenizer } from '../preprocessing/tokenizer.js';

export class StabilityAnalyzer {
  /**
   * Compare two model checkpoints
   *
   * @param {string} checkpoint1Path - Path to earlier checkpoint
   * @param {string} checkpoint2Path - Path to later checkpoint
   * @param {Array<string>} probeWords - Words to analyze (optional)
   * @returns {Object} - Stability analysis results
   */
  static compareCheckpoints(checkpoint1Path, checkpoint2Path, probeWords = null) {
    console.log('\n' + '='.repeat(70));
    console.log('STABILITY ANALYSIS: Comparing Checkpoints');
    console.log('='.repeat(70) + '\n');

    // Load both models
    console.log('Loading checkpoints...');
    const model1 = ModelPersistence.loadModel(checkpoint1Path);
    const model2 = ModelPersistence.loadModel(checkpoint2Path);

    console.log(`  Checkpoint 1: ${checkpoint1Path}`);
    console.log(`  Checkpoint 2: ${checkpoint2Path}\n`);

    // Reconstruct trainers
    const trainer1 = new CSSTrainer(model1.config);
    trainer1.initialize(model1.modelData.vocabSize);
    trainer1.importModel(model1.modelData);

    const trainer2 = new CSSTrainer(model2.config);
    trainer2.initialize(model2.modelData.vocabSize);
    trainer2.importModel(model2.modelData);

    // Determine probe words
    let wordsToAnalyze = probeWords;
    if (!wordsToAnalyze) {
      // Default: select some high-frequency words
      const wordFreqs = Array.from(model1.wordFreq.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 20)
        .map(([word]) => word);

      wordsToAnalyze = wordFreqs;
      console.log(`Analyzing top ${wordsToAnalyze.length} frequent words\n`);
    } else {
      console.log(`Analyzing ${wordsToAnalyze.length} probe words\n`);
    }

    // Compare spectra for each word
    const results = [];

    for (const word of wordsToAnalyze) {
      const wordId1 = model1.vocab.get(word);
      const wordId2 = model2.vocab.get(word);

      if (wordId1 === undefined || wordId2 === undefined) {
        console.log(`  ⚠️  "${word}" not in both checkpoints, skipping`);
        continue;
      }

      const comparison = this.compareWordSpectra(
        word,
        wordId1,
        trainer1,
        wordId2,
        trainer2
      );

      results.push(comparison);
    }

    // Aggregate statistics
    const stats = this.computeStabilityStats(results);

    // Print summary
    console.log('\n' + '='.repeat(70));
    console.log('STABILITY SUMMARY');
    console.log('='.repeat(70) + '\n');

    console.log(`Words analyzed: ${results.length}`);
    console.log(`\nSpectral similarity (0=changed completely, 1=unchanged):`);
    console.log(`  Mean:   ${stats.meanSimilarity.toFixed(4)}`);
    console.log(`  Median: ${stats.medianSimilarity.toFixed(4)}`);
    console.log(`  Min:    ${stats.minSimilarity.toFixed(4)} (word: "${stats.mostChanged}")`);
    console.log(`  Max:    ${stats.maxSimilarity.toFixed(4)} (word: "${stats.mostStable}")`);

    console.log(`\nSparsity change:`);
    console.log(`  Mean:   ${stats.meanSparsityChange.toFixed(2)} frequencies`);
    console.log(`  Median: ${stats.medianSparsityChange.toFixed(2)} frequencies`);

    console.log(`\nFrequency turnover:`);
    console.log(`  Mean:   ${stats.meanTurnover.toFixed(2)}%`);
    console.log(`  Median: ${stats.medianTurnover.toFixed(2)}%`);

    // Verdict
    console.log('\n' + '-'.repeat(70));
    console.log('VERDICT:');
    console.log('-'.repeat(70));

    const verdict = this.computeStabilityVerdict(stats);
    console.log(verdict.message);
    console.log('-'.repeat(70) + '\n');

    return {
      checkpoint1: checkpoint1Path,
      checkpoint2: checkpoint2Path,
      results,
      stats,
      verdict
    };
  }

  /**
   * Compare spectra for a single word between two checkpoints
   */
  static compareWordSpectra(word, wordId1, trainer1, wordId2, trainer2) {
    const spec1 = trainer1.spectralWord.getSpectrum(wordId1);
    const spec2 = trainer2.spectralWord.getSpectrum(wordId2);

    // Convert to dense for comparison
    const vec1 = trainer1.spectralWord.toDenseVector(wordId1);
    const vec2 = trainer2.spectralWord.toDenseVector(wordId2);

    // 1. Spectral similarity (cosine similarity)
    const similarity = this.cosineSimilarity(vec1, vec2);

    // 2. Sparsity change
    const sparsity1 = spec1.frequencies.length;
    const sparsity2 = spec2.frequencies.length;
    const sparsityChange = Math.abs(sparsity2 - sparsity1);

    // 3. Frequency turnover (how many frequencies changed)
    const freqSet1 = new Set(spec1.frequencies);
    const freqSet2 = new Set(spec2.frequencies);

    const kept = [...freqSet1].filter(f => freqSet2.has(f)).length;
    const added = [...freqSet2].filter(f => !freqSet1.has(f)).length;
    const removed = [...freqSet1].filter(f => !freqSet2.has(f)).length;

    const totalUnique = new Set([...freqSet1, ...freqSet2]).size;
    const turnoverRate = totalUnique > 0 ? ((added + removed) / totalUnique) * 100 : 0;

    // 4. Amplitude change (for kept frequencies)
    let amplitudeChange = 0;
    if (kept > 0) {
      for (const freq of freqSet1) {
        if (freqSet2.has(freq)) {
          const idx1 = spec1.frequencies.indexOf(freq);
          const idx2 = spec2.frequencies.indexOf(freq);
          const amp1 = spec1.amplitudes[idx1];
          const amp2 = spec2.amplitudes[idx2];
          amplitudeChange += Math.abs(amp2 - amp1);
        }
      }
      amplitudeChange /= kept;
    }

    return {
      word,
      similarity,
      sparsity1,
      sparsity2,
      sparsityChange,
      kept,
      added,
      removed,
      turnoverRate,
      amplitudeChange
    };
  }

  /**
   * Cosine similarity between two vectors
   */
  static cosineSimilarity(vec1, vec2) {
    let dot = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < vec1.length; i++) {
      dot += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }

    norm1 = Math.sqrt(norm1);
    norm2 = Math.sqrt(norm2);

    if (norm1 === 0 || norm2 === 0) return 0;

    return dot / (norm1 * norm2);
  }

  /**
   * Compute aggregate statistics
   */
  static computeStabilityStats(results) {
    const similarities = results.map(r => r.similarity);
    const sparsityChanges = results.map(r => r.sparsityChange);
    const turnovers = results.map(r => r.turnoverRate);

    // Sort for median
    const sortedSimilarities = [...similarities].sort((a, b) => a - b);
    const sortedSparsityChanges = [...sparsityChanges].sort((a, b) => a - b);
    const sortedTurnovers = [...turnovers].sort((a, b) => a - b);

    const median = arr => arr[Math.floor(arr.length / 2)];
    const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;

    // Find most changed and most stable words
    const mostChangedIdx = similarities.indexOf(Math.min(...similarities));
    const mostStableIdx = similarities.indexOf(Math.max(...similarities));

    return {
      meanSimilarity: mean(similarities),
      medianSimilarity: median(sortedSimilarities),
      minSimilarity: Math.min(...similarities),
      maxSimilarity: Math.max(...similarities),
      mostChanged: results[mostChangedIdx].word,
      mostStable: results[mostStableIdx].word,

      meanSparsityChange: mean(sparsityChanges),
      medianSparsityChange: median(sortedSparsityChanges),

      meanTurnover: mean(turnovers),
      medianTurnover: median(sortedTurnovers)
    };
  }

  /**
   * Determine training stability verdict
   */
  static computeStabilityVerdict(stats) {
    const { meanSimilarity, meanTurnover } = stats;

    let status = '';
    let message = '';

    if (meanSimilarity > 0.95) {
      status = 'FROZEN';
      message = `⚠️  SPECTRA FROZEN (similarity: ${meanSimilarity.toFixed(3)})\n` +
                `   Spectra barely changing between checkpoints\n` +
                `   Possible causes:\n` +
                `   - Learning rate too low\n` +
                `   - Sparsity penalty too aggressive\n` +
                `   - Already converged (check loss)\n` +
                `   Recommendation: Increase learning rate or reduce sparsity penalty`;
    } else if (meanSimilarity < 0.3) {
      status = 'UNSTABLE';
      message = `⚠️  UNSTABLE TRAINING (similarity: ${meanSimilarity.toFixed(3)})\n` +
                `   Spectra changing wildly between checkpoints\n` +
                `   Possible causes:\n` +
                `   - Learning rate too high\n` +
                `   - Training not converging\n` +
                `   - Model oscillating\n` +
                `   Recommendation: Reduce learning rate or add momentum`;
    } else if (meanSimilarity < 0.6) {
      status = 'LEARNING';
      message = `✅ ACTIVELY LEARNING (similarity: ${meanSimilarity.toFixed(3)})\n` +
                `   Spectra changing at healthy rate\n` +
                `   Frequency turnover: ${meanTurnover.toFixed(1)}%\n` +
                `   This is expected in early training\n` +
                `   Continue training and check stability later`;
    } else {
      status = 'STABILIZING';
      message = `✅ STABILIZING (similarity: ${meanSimilarity.toFixed(3)})\n` +
                `   Spectra converging but still refining\n` +
                `   Frequency turnover: ${meanTurnover.toFixed(1)}%\n` +
                `   This is expected in late training\n` +
                `   Model approaching convergence`;
    }

    return { status, message };
  }

  /**
   * Print detailed comparison for a single word
   */
  static printWordComparison(comparison) {
    console.log(`\nWord: "${comparison.word}"`);
    console.log(`  Similarity: ${comparison.similarity.toFixed(4)}`);
    console.log(`  Sparsity: ${comparison.sparsity1} → ${comparison.sparsity2} (Δ${comparison.sparsityChange})`);
    console.log(`  Frequencies: ${comparison.kept} kept, ${comparison.added} added, ${comparison.removed} removed`);
    console.log(`  Turnover rate: ${comparison.turnoverRate.toFixed(1)}%`);

    if (comparison.kept > 0) {
      console.log(`  Amplitude change (kept freqs): ${comparison.amplitudeChange.toFixed(4)}`);
    }
  }

  /**
   * Analyze stability over multiple checkpoints (time series)
   */
  static analyzeStabilityTimeSeries(checkpointPaths, probeWords = null) {
    console.log('\n' + '█'.repeat(70));
    console.log('STABILITY TIME SERIES ANALYSIS');
    console.log('█'.repeat(70) + '\n');

    console.log(`Analyzing ${checkpointPaths.length} checkpoints\n`);

    const timeSeriesResults = [];

    // Compare consecutive checkpoints
    for (let i = 0; i < checkpointPaths.length - 1; i++) {
      console.log(`\nComparing checkpoint ${i + 1} → ${i + 2}...`);

      const result = this.compareCheckpoints(
        checkpointPaths[i],
        checkpointPaths[i + 1],
        probeWords
      );

      timeSeriesResults.push({
        step: i + 1,
        checkpoint1: checkpointPaths[i],
        checkpoint2: checkpointPaths[i + 1],
        stats: result.stats,
        verdict: result.verdict
      });
    }

    // Print time series summary
    console.log('\n\n' + '█'.repeat(70));
    console.log('TIME SERIES SUMMARY');
    console.log('█'.repeat(70) + '\n');

    console.log('Similarity over time (higher = more stable):');
    console.log('Step | Mean Similarity | Turnover | Status');
    console.log('-'.repeat(60));

    for (const result of timeSeriesResults) {
      const sim = result.stats.meanSimilarity.toFixed(3);
      const turn = result.stats.meanTurnover.toFixed(1);
      const status = result.verdict.status;

      console.log(`${result.step}→${result.step + 1}  | ${sim}           | ${turn}%     | ${status}`);
    }

    // Check for trends
    console.log('\n' + '-'.repeat(70));
    console.log('TRENDS:');
    console.log('-'.repeat(70));

    const similarities = timeSeriesResults.map(r => r.stats.meanSimilarity);

    if (similarities.length > 2) {
      const early = similarities[0];
      const late = similarities[similarities.length - 1];

      if (late > early + 0.2) {
        console.log('✅ Spectra are STABILIZING over time (good)');
      } else if (late < early - 0.2) {
        console.log('⚠️  Spectra are becoming LESS stable over time (concerning)');
      } else {
        console.log('→ Stability relatively CONSTANT over time');
      }
    }

    console.log('-'.repeat(70) + '\n');

    return timeSeriesResults;
  }
}
