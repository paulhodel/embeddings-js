/**
 * PolysemyAnalyzer - Analyze sense separation for polysemous words
 *
 * This tool helps verify if CSS is learning multiple senses by:
 * 1. Collecting contexts where a word appears
 * 2. Building context spectra (without the target word)
 * 3. Clustering context spectra
 * 4. Checking if word's frequencies align with cluster centroids
 */

export class PolysemyAnalyzer {
  constructor(trainer, tokenizer) {
    this.trainer = trainer;
    this.tokenizer = tokenizer;
    this.spectralWord = trainer.spectralWord;
  }

  /**
   * Analyze sense separation for a polysemous word
   *
   * @param {string} word - The word to analyze (e.g., "bank")
   * @param {Array<Array<number>>} corpus - Corpus as word ID sequences
   * @param {number} numClusters - Expected number of senses (default: 2)
   * @returns {Object} - Analysis results
   */
  analyzeSenseSeparation(word, corpus, numClusters = 2) {
    console.log('\n' + '='.repeat(70));
    console.log(`POLYSEMY ANALYSIS: "${word}"`);
    console.log('='.repeat(70) + '\n');

    const wordId = this.tokenizer.wordToId(word);

    if (wordId === undefined) {
      console.log(`❌ Word "${word}" not in vocabulary`);
      return null;
    }

    // Step 1: Collect all contexts where word appears
    console.log('Step 1: Collecting contexts...');
    const contexts = this.collectContexts(wordId, corpus);
    console.log(`  Found ${contexts.length} occurrences of "${word}"\n`);

    if (contexts.length < numClusters * 2) {
      console.log(`⚠️  Too few occurrences (${contexts.length}) for ${numClusters} clusters`);
      return null;
    }

    // Step 2: Build context spectra (without target word)
    console.log('Step 2: Building context spectra...');
    const contextSpectra = this.buildContextSpectra(contexts, wordId);
    console.log(`  Built ${contextSpectra.length} context spectra\n`);

    // Step 3: Cluster context spectra
    console.log(`Step 3: Clustering into ${numClusters} sense clusters...`);
    const clusters = this.clusterContexts(contextSpectra, numClusters);

    // Print cluster information
    for (let i = 0; i < clusters.centroids.length; i++) {
      console.log(`\n  Cluster ${i + 1}: ${clusters.assignments.filter(a => a === i).length} contexts`);
      console.log(`    Top frequencies: ${this.getTopFrequencies(clusters.centroids[i], 5)}`);

      // Show sample contexts from this cluster
      const clusterContexts = contexts.filter((_, idx) => clusters.assignments[idx] === i);
      const samples = clusterContexts.slice(0, 3);

      console.log(`    Sample contexts:`);
      for (const ctx of samples) {
        const contextWords = ctx.context.map(id => this.tokenizer.idToWord.get(id)).filter(w => w);
        console.log(`      - [...${contextWords.slice(0, 5).join(' ')}...]`);
      }
    }

    // Step 4: Analyze target word's spectrum
    console.log('\n\nStep 4: Analyzing target word spectrum...');
    const wordSpectrum = this.spectralWord.getSpectrum(wordId);
    console.log(`  "${word}" has ${wordSpectrum.frequencies.length} active frequencies`);

    const dominantFreqs = this.trainer.getWordSpectrum(wordId, 10);
    console.log(`\n  Dominant frequencies:`);
    dominantFreqs.forEach((freq, idx) => {
      console.log(`    ${idx + 1}. Freq=${freq.frequency}, Amp=${freq.amplitude.toFixed(4)}`);
    });

    // Step 5: Check alignment with cluster centroids
    console.log('\n\nStep 5: Checking sense-cluster alignment...');
    const alignment = this.checkAlignment(wordSpectrum, clusters.centroids);

    for (let i = 0; i < clusters.centroids.length; i++) {
      console.log(`\n  Sense ${i + 1} (Cluster ${i + 1}):`);
      console.log(`    Alignment score: ${alignment.scores[i].toFixed(4)}`);
      console.log(`    Overlapping frequencies: ${alignment.overlaps[i].join(', ') || 'none'}`);
    }

    // Step 6: Verdict
    console.log('\n\n' + '-'.repeat(70));
    console.log('VERDICT:');
    console.log('-'.repeat(70));

    const verdict = this.computeVerdict(wordSpectrum, clusters, alignment);
    console.log(verdict.message);
    console.log('-'.repeat(70) + '\n');

    return {
      word,
      wordId,
      contexts,
      contextSpectra,
      clusters,
      wordSpectrum,
      alignment,
      verdict
    };
  }

  /**
   * Collect all contexts where target word appears
   */
  collectContexts(wordId, corpus) {
    const contexts = [];
    const windowSize = this.trainer.config.windowSize || 3;

    for (const doc of corpus) {
      for (let i = 0; i < doc.length; i++) {
        if (doc[i] === wordId) {
          // Extract context window (excluding target word)
          const context = [];

          for (let j = Math.max(0, i - windowSize);
               j <= Math.min(doc.length - 1, i + windowSize);
               j++) {
            if (j !== i) {
              context.push(doc[j]);
            }
          }

          if (context.length > 0) {
            contexts.push({ position: i, context, document: doc });
          }
        }
      }
    }

    return contexts;
  }

  /**
   * Build context spectra (sum of context word spectra, excluding target)
   */
  buildContextSpectra(contexts, excludeWordId) {
    const spectra = [];

    for (const ctx of contexts) {
      const contextSpectrum = new Array(this.trainer.config.frequencyDim).fill(0);

      for (const contextWordId of ctx.context) {
        if (contextWordId === excludeWordId) continue;

        const wordVec = this.spectralWord.toDenseVector(contextWordId);

        for (let i = 0; i < wordVec.length; i++) {
          contextSpectrum[i] += wordVec[i];
        }
      }

      // Normalize
      const norm = Math.sqrt(contextSpectrum.reduce((sum, val) => sum + val * val, 0));
      if (norm > 0) {
        for (let i = 0; i < contextSpectrum.length; i++) {
          contextSpectrum[i] /= norm;
        }
      }

      spectra.push(contextSpectrum);
    }

    return spectra;
  }

  /**
   * Simple k-means clustering on context spectra
   */
  clusterContexts(spectra, k, maxIters = 20) {
    if (spectra.length < k) {
      throw new Error(`Not enough spectra (${spectra.length}) for ${k} clusters`);
    }

    // Initialize centroids randomly
    const centroids = [];
    const usedIndices = new Set();

    for (let i = 0; i < k; i++) {
      let idx;
      do {
        idx = Math.floor(Math.random() * spectra.length);
      } while (usedIndices.has(idx));

      usedIndices.add(idx);
      centroids.push([...spectra[idx]]);
    }

    let assignments = new Array(spectra.length).fill(0);

    // K-means iterations
    for (let iter = 0; iter < maxIters; iter++) {
      // Assign to nearest centroid
      let changed = 0;

      for (let i = 0; i < spectra.length; i++) {
        let minDist = Infinity;
        let bestCluster = 0;

        for (let j = 0; j < k; j++) {
          const dist = this.euclideanDistance(spectra[i], centroids[j]);
          if (dist < minDist) {
            minDist = dist;
            bestCluster = j;
          }
        }

        if (assignments[i] !== bestCluster) {
          changed++;
          assignments[i] = bestCluster;
        }
      }

      // Update centroids
      const newCentroids = Array(k).fill(null).map(() =>
        new Array(spectra[0].length).fill(0)
      );
      const counts = new Array(k).fill(0);

      for (let i = 0; i < spectra.length; i++) {
        const cluster = assignments[i];
        counts[cluster]++;

        for (let j = 0; j < spectra[i].length; j++) {
          newCentroids[cluster][j] += spectra[i][j];
        }
      }

      for (let i = 0; i < k; i++) {
        if (counts[i] > 0) {
          for (let j = 0; j < newCentroids[i].length; j++) {
            centroids[i][j] = newCentroids[i][j] / counts[i];
          }
        }
      }

      if (changed === 0) {
        console.log(`  Converged after ${iter + 1} iterations`);
        break;
      }
    }

    return { centroids, assignments };
  }

  /**
   * Calculate Euclidean distance between two vectors
   */
  euclideanDistance(vec1, vec2) {
    let sum = 0;
    for (let i = 0; i < vec1.length; i++) {
      const diff = vec1[i] - vec2[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  /**
   * Get top K frequencies from a spectrum
   */
  getTopFrequencies(spectrum, k) {
    const freqAmpPairs = spectrum
      .map((amp, freq) => ({ freq, amp }))
      .filter(p => p.amp > 0.001)
      .sort((a, b) => b.amp - a.amp)
      .slice(0, k);

    return freqAmpPairs.map(p => `${p.freq}(${p.amp.toFixed(3)})`).join(', ');
  }

  /**
   * Check alignment between word spectrum and cluster centroids
   */
  checkAlignment(wordSpectrum, centroids) {
    const scores = [];
    const overlaps = [];

    const wordDense = new Array(this.trainer.config.frequencyDim).fill(0);
    for (let i = 0; i < wordSpectrum.frequencies.length; i++) {
      wordDense[wordSpectrum.frequencies[i]] = wordSpectrum.amplitudes[i];
    }

    for (const centroid of centroids) {
      // Cosine similarity
      let dotProduct = 0;
      let norm1 = 0;
      let norm2 = 0;

      const overlapFreqs = [];

      for (let i = 0; i < wordDense.length; i++) {
        dotProduct += wordDense[i] * centroid[i];
        norm1 += wordDense[i] * wordDense[i];
        norm2 += centroid[i] * centroid[i];

        if (wordDense[i] > 0.01 && centroid[i] > 0.01) {
          overlapFreqs.push(i);
        }
      }

      const similarity = (norm1 > 0 && norm2 > 0) ?
        dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2)) : 0;

      scores.push(similarity);
      overlaps.push(overlapFreqs);
    }

    return { scores, overlaps };
  }

  /**
   * Compute verdict on polysemy emergence
   */
  computeVerdict(wordSpectrum, clusters, alignment) {
    const numActiveFreqs = wordSpectrum.frequencies.length;
    const numClusters = clusters.centroids.length;
    const maxAlignment = Math.max(...alignment.scores);
    const minAlignment = Math.min(...alignment.scores);
    const alignmentSpread = maxAlignment - minAlignment;

    let message = '';
    let status = '';

    if (numActiveFreqs === 1) {
      status = 'NO_POLYSEMY';
      message = `❌ POLYSEMY NOT EMERGED\n` +
                `   Word has only 1 active frequency\n` +
                `   Contexts cluster into ${numClusters} groups, but word doesn't separate senses`;
    } else if (numActiveFreqs < numClusters) {
      status = 'PARTIAL';
      message = `⚠️  PARTIAL POLYSEMY\n` +
                `   Word has ${numActiveFreqs} frequencies but ${numClusters} context clusters\n` +
                `   May need more training or more data`;
    } else if (alignmentSpread < 0.1) {
      status = 'WEAK';
      message = `⚠️  WEAK SENSE SEPARATION\n` +
                `   Word has ${numActiveFreqs} frequencies\n` +
                `   But alignment scores are similar (spread: ${alignmentSpread.toFixed(3)})\n` +
                `   Senses may not be well-differentiated`;
    } else {
      status = 'SUCCESS';
      message = `✅ POLYSEMY DETECTED!\n` +
                `   Word has ${numActiveFreqs} active frequencies\n` +
                `   ${numClusters} distinct context clusters found\n` +
                `   Alignment spread: ${alignmentSpread.toFixed(3)}\n` +
                `   Frequencies appear to separate senses`;
    }

    return { status, message, numActiveFreqs, alignmentSpread };
  }
}
