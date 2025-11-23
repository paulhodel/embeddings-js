/**
 * Polysemy Analysis Script
 *
 * Usage:
 *   node src/analyze_polysemy.js <snapshot_file>
 *
 * Analyzes sense separation for polysemous words like "bank" using a trained model snapshot
 */

import fs from 'fs';
import path from 'path';
import {execSync} from 'child_process';

const CONFIG = {
    parquetDir: './data/parquet',
    snapshotDir: './data/snapshots',
    frequencyDim: 256,
    windowSize: 2,
    maxContextsToAnalyze: 100,  // Limit contexts analyzed per word
    multiPeakThreshold: 0.3,    // Threshold for considering a peak "dominant"
    monosemousWords: ['oxygen', 'hydrogen', 'carbon', 'nitrogen', 'helium']  // For comparison
};

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Load snapshot file
 */
function loadSnapshot(filepath) {
    if (!fs.existsSync(filepath)) {
        console.error(`Snapshot file not found: ${filepath}`);
        process.exit(1);
    }

    console.log(`Loading snapshot: ${filepath}\n`);
    const data = fs.readFileSync(filepath, 'utf-8');
    return JSON.parse(data);
}

/**
 * Convert sparse spectrum to dense complex vector
 */
function spectrumToDense(spectrum, frequencyDim) {
    const dense = new Float32Array(frequencyDim);

    if (!spectrum || !spectrum.frequencies) return dense;

    for (let i = 0; i < spectrum.frequencies.length; i++) {
        const freq = spectrum.frequencies[i];
        const amp = spectrum.amplitudes[i];

        dense[freq] = amp;  // Real-only (no phases in prototype)
    }

    return dense;
}

/**
 * Compute compatibility score (dot product for real-only prototype)
 */
function computeScore(spectrum1Dense, spectrum2Dense) {
    let score = 0;

    for (let i = 0; i < spectrum1Dense.length; i++) {
        score += spectrum1Dense[i] * spectrum2Dense[i];
    }

    return score;
}

/**
 * Load parquet file and extract documents
 */
function loadParquetFile(filepath) {
    console.log(`Loading parquet: ${filepath}`);

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
 * Extract contexts for a target word from corpus
 */
function extractContexts(targetWord, snapshot, documents) {
    const contexts = [];
    const windowSize = CONFIG.windowSize;

    // Find target word in vocabulary
    const wordObj = snapshot.vocabulary.find(w => w.word === targetWord);
    if (!wordObj) {
        console.log(`Word "${targetWord}" not found in vocabulary`);
        return [];
    }

    // Create word lookup for fast access
    const wordLookup = new Map();
    for (const word of snapshot.vocabulary) {
        wordLookup.set(word.word, word);
    }

    // Process each document
    for (const content of documents) {
        const tokens = [];
        let token = '';

        // Tokenize
        for (let i = 0; i < content.length; i++) {
            const char = content[i];
            const code = char.charCodeAt(0);

            const isAlpha = (code >= 65 && code <= 90) || (code >= 97 && code <= 122);
            const isDigit = (code >= 48 && code <= 57);

            if (isAlpha || isDigit) {
                token += isAlpha && code <= 90 ? char.toLowerCase() : char;
            } else {
                if (token.length > 0) {
                    tokens.push(token);
                    token = '';
                }
            }
        }
        if (token.length > 0) tokens.push(token);

        // Extract contexts where target word appears
        for (let i = 0; i < tokens.length; i++) {
            if (tokens[i] === targetWord) {
                const start = Math.max(0, i - windowSize);
                const end = Math.min(tokens.length, i + windowSize + 1);

                const contextWords = [];
                for (let j = start; j < end; j++) {
                    if (j !== i) {
                        const w = wordLookup.get(tokens[j]);
                        if (w && w.spectrum) {
                            contextWords.push(w);
                        }
                    }
                }

                if (contextWords.length > 0) {
                    contexts.push({
                        position: i,
                        contextWords: contextWords,
                        tokens: tokens.slice(start, end)
                    });

                    if (contexts.length >= CONFIG.maxContextsToAnalyze) {
                        return contexts;
                    }
                }
            }
        }
    }

    return contexts;
}

/**
 * Build average context spectrum from context words
 */
function buildContextSpectrum(contextWords, frequencyDim) {
    const contextDense = new Float32Array(frequencyDim * 2);

    for (const wordObj of contextWords) {
        const wordDense = spectrumToDense(wordObj.spectrum, frequencyDim);
        for (let i = 0; i < contextDense.length; i++) {
            contextDense[i] += wordDense[i];
        }
    }

    if (contextWords.length > 0) {
        for (let i = 0; i < contextDense.length; i++) {
            contextDense[i] /= contextWords.length;
        }
    }

    return contextDense;
}

/**
 * Simple K-means clustering on context spectra
 */
function clusterContexts(contextSpectra, k = 2) {
    if (contextSpectra.length < k) return null;

    const dim = contextSpectra[0].length;

    // Initialize centroids randomly
    const centroids = [];
    const usedIndices = new Set();
    while (centroids.length < k) {
        const idx = Math.floor(Math.random() * contextSpectra.length);
        if (!usedIndices.has(idx)) {
            usedIndices.add(idx);
            centroids.push(new Float32Array(contextSpectra[idx]));
        }
    }

    let assignments = new Array(contextSpectra.length).fill(0);
    let changed = true;
    let iterations = 0;
    const maxIterations = 50;

    while (changed && iterations < maxIterations) {
        changed = false;
        iterations++;

        // Assign to nearest centroid
        for (let i = 0; i < contextSpectra.length; i++) {
            let minDist = Infinity;
            let bestCluster = 0;

            for (let c = 0; c < k; c++) {
                let dist = 0;
                for (let d = 0; d < dim; d++) {
                    const diff = contextSpectra[i][d] - centroids[c][d];
                    dist += diff * diff;
                }

                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = c;
                }
            }

            if (assignments[i] !== bestCluster) {
                assignments[i] = bestCluster;
                changed = true;
            }
        }

        // Update centroids
        const counts = new Array(k).fill(0);
        for (let c = 0; c < k; c++) {
            centroids[c].fill(0);
        }

        for (let i = 0; i < contextSpectra.length; i++) {
            const cluster = assignments[i];
            counts[cluster]++;
            for (let d = 0; d < dim; d++) {
                centroids[cluster][d] += contextSpectra[i][d];
            }
        }

        for (let c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (let d = 0; d < dim; d++) {
                    centroids[c][d] /= counts[c];
                }
            }
        }
    }

    return {assignments, centroids};
}

/**
 * TEST 1: Analyze spectrum for multiple dominant peaks
 */
function analyzeSpectralPeaks(spectrum, threshold = CONFIG.multiPeakThreshold) {
    if (!spectrum || !spectrum.amplitudes) {
        return { peaks: [], peakCount: 0, maxAmplitude: 0, isPotentiallyPolysemous: false };
    }

    // Find peaks above threshold
    const peaks = [];
    for (let i = 0; i < spectrum.amplitudes.length; i++) {
        if (spectrum.amplitudes[i] >= threshold) {
            peaks.push({
                frequency: spectrum.frequencies[i],
                amplitude: spectrum.amplitudes[i]
            });
        }
    }

    // Sort by amplitude
    peaks.sort((a, b) => b.amplitude - a.amplitude);

    const maxAmplitude = peaks.length > 0 ? peaks[0].amplitude : 0;
    const isPotentiallyPolysemous = peaks.length >= 2;

    return {
        peaks,
        peakCount: peaks.length,
        maxAmplitude,
        isPotentiallyPolysemous,
        allAmplitudes: spectrum.amplitudes
    };
}

/**
 * TEST 2: Load all snapshots and track word evolution
 */
function loadAllSnapshots() {
    if (!fs.existsSync(CONFIG.snapshotDir)) {
        return [];
    }

    const snapshotFiles = fs.readdirSync(CONFIG.snapshotDir)
        .filter(f => f.startsWith('snapshot_') && f.endsWith('.json'))
        .sort();

    const snapshots = [];
    for (const file of snapshotFiles) {
        try {
            const filepath = path.join(CONFIG.snapshotDir, file);
            const data = fs.readFileSync(filepath, 'utf-8');
            snapshots.push(JSON.parse(data));
        } catch (err) {
            console.error(`Failed to load ${file}: ${err.message}`);
        }
    }

    return snapshots;
}

/**
 * TEST 2: Track frequency amplitude evolution across snapshots
 */
function trackFrequencyEvolution(word, snapshots) {
    const evolution = [];

    for (const snapshot of snapshots) {
        const wordObj = snapshot.vocabulary.find(w => w.word === word);
        if (!wordObj || !wordObj.spectrum) {
            evolution.push({
                snapshotNumber: snapshot.snapshotNumber,
                documents: snapshot.trainingProgress.totalDocuments,
                spectrum: null,
                frequencies: []
            });
            continue;
        }

        const freqMap = new Map();
        for (let i = 0; i < wordObj.spectrum.frequencies.length; i++) {
            freqMap.set(wordObj.spectrum.frequencies[i], {
                amplitude: wordObj.spectrum.amplitudes[i]
            });
        }

        evolution.push({
            snapshotNumber: snapshot.snapshotNumber,
            documents: snapshot.trainingProgress.totalDocuments,
            spectrum: wordObj.spectrum,
            frequencies: freqMap
        });
    }

    return evolution;
}

/**
 * TEST 4: Analyze mutual reinforcement between frequency groups
 */
function analyzeMutualReinforcement(targetWord, snapshot, contexts, clustering) {
    const wordObj = snapshot.vocabulary.find(w => w.word === targetWord);
    if (!wordObj || !wordObj.spectrum) return null;

    // Get top frequencies for each cluster centroid
    const clusterFreqs = [];
    for (let c = 0; c < clustering.centroids.length; c++) {
        const centroid = clustering.centroids[c];
        const topFreqs = [];

        // Convert dense centroid back to frequency list
        for (let freq = 0; freq < CONFIG.frequencyDim; freq++) {
            const real = centroid[freq * 2];
            const imag = centroid[freq * 2 + 1];
            const amp = Math.sqrt(real * real + imag * imag);

            if (amp > 0.01) {
                topFreqs.push({ frequency: freq, amplitude: amp });
            }
        }

        topFreqs.sort((a, b) => b.amplitude - a.amplitude);
        clusterFreqs.push(topFreqs.slice(0, 5));  // Top 5 per cluster
    }

    // Calculate overlap between clusters
    const cluster0Freqs = new Set(clusterFreqs[0].map(f => f.frequency));
    const cluster1Freqs = new Set(clusterFreqs[1].map(f => f.frequency));

    const overlap = [...cluster0Freqs].filter(f => cluster1Freqs.has(f));
    const overlapRatio = overlap.length / Math.max(cluster0Freqs.size, cluster1Freqs.size);

    return {
        clusterFrequencies: clusterFreqs,
        overlap: overlap,
        overlapRatio: overlapRatio,
        isSeparated: overlapRatio < 0.3  // Less than 30% overlap = good separation
    };
}

/**
 * TEST 5: Test spectrum stability under context substitution
 */
function testContextSubstitution(targetWord, snapshot, contexts, clustering) {
    const wordObj = snapshot.vocabulary.find(w => w.word === targetWord);
    if (!wordObj || !wordObj.spectrum) return null;

    const wordDense = spectrumToDense(wordObj.spectrum, CONFIG.frequencyDim);

    // For each cluster, compute average similarity
    const clusterSimilarities = [];

    for (let c = 0; c < clustering.centroids.length; c++) {
        const clusterContexts = contexts.filter((_, idx) =>
            clustering.assignments[idx] === c
        );

        let totalSimilarity = 0;
        for (const ctx of clusterContexts) {
            const contextSpec = buildContextSpectrum(ctx.contextWords, CONFIG.frequencyDim);
            const similarity = computeScore(wordDense, contextSpec);
            totalSimilarity += similarity;
        }

        const avgSimilarity = clusterContexts.length > 0 ? totalSimilarity / clusterContexts.length : 0;
        clusterSimilarities.push({
            cluster: c,
            avgSimilarity: avgSimilarity,
            count: clusterContexts.length
        });
    }

    // Check if similarities are significantly different
    const diff = Math.abs(clusterSimilarities[0].avgSimilarity - clusterSimilarities[1].avgSimilarity);
    const avgSim = (clusterSimilarities[0].avgSimilarity + clusterSimilarities[1].avgSimilarity) / 2;
    const relativeDiff = avgSim > 0 ? diff / avgSim : 0;

    return {
        clusterSimilarities,
        isDifferentiated: relativeDiff > 0.2  // 20% difference = good differentiation
    };
}

/**
 * Calculate silhouette score for clustering quality
 */
function calculateSilhouette(contextSpectra, assignments) {
    const k = Math.max(...assignments) + 1;
    let totalScore = 0;

    for (let i = 0; i < contextSpectra.length; i++) {
        const cluster = assignments[i];

        // Average distance to points in same cluster (a)
        let a = 0;
        let countA = 0;
        for (let j = 0; j < contextSpectra.length; j++) {
            if (i !== j && assignments[j] === cluster) {
                let dist = 0;
                for (let d = 0; d < contextSpectra[i].length; d++) {
                    const diff = contextSpectra[i][d] - contextSpectra[j][d];
                    dist += diff * diff;
                }
                a += Math.sqrt(dist);
                countA++;
            }
        }
        a = countA > 0 ? a / countA : 0;

        // Average distance to nearest other cluster (b)
        let b = Infinity;
        for (let c = 0; c < k; c++) {
            if (c !== cluster) {
                let avgDist = 0;
                let countB = 0;
                for (let j = 0; j < contextSpectra.length; j++) {
                    if (assignments[j] === c) {
                        let dist = 0;
                        for (let d = 0; d < contextSpectra[i].length; d++) {
                            const diff = contextSpectra[i][d] - contextSpectra[j][d];
                            dist += diff * diff;
                        }
                        avgDist += Math.sqrt(dist);
                        countB++;
                    }
                }
                if (countB > 0) {
                    avgDist /= countB;
                    b = Math.min(b, avgDist);
                }
            }
        }

        const silhouette = (b - a) / Math.max(a, b);
        totalScore += silhouette;
    }

    return totalScore / contextSpectra.length;
}

// ============================================
// MAIN ANALYSIS
// ============================================

async function analyzePolysemy() {
    console.log('\n' + '█'.repeat(70));
    console.log('COMPREHENSIVE POLYSEMY ANALYSIS');
    console.log('█'.repeat(70) + '\n');

    // Parse arguments
    const args = process.argv.slice(2);
    if (args.length === 0) {
        console.error('Usage: node src/analyze_polysemy.js <snapshot_file> [target_word] [comparison_word]');
        console.error('Example: node src/analyze_polysemy.js ./data/snapshots/snapshot_0001_docs5000.json bank oxygen');
        console.error('\nOr analyze using all snapshots:');
        console.error('node src/analyze_polysemy.js --all bank');
        process.exit(1);
    }

    const useAllSnapshots = args[0] === '--all';
    const snapshotFile = useAllSnapshots ? null : args[0];
    const targetWord = useAllSnapshots ? args[1] : (args[1] || 'bank');
    const comparisonWord = useAllSnapshots ? (args[2] || 'oxygen') : (args[2] || 'oxygen');

    let snapshot;
    let allSnapshots = [];

    if (useAllSnapshots) {
        console.log('Loading all snapshots for temporal analysis...\n');
        allSnapshots = loadAllSnapshots();
        if (allSnapshots.length === 0) {
            console.error('No snapshots found in:', CONFIG.snapshotDir);
            process.exit(1);
        }
        snapshot = allSnapshots[allSnapshots.length - 1];  // Use latest for main analysis
        console.log(`Found ${allSnapshots.length} snapshots\n`);
    } else {
        snapshot = loadSnapshot(snapshotFile);
    }

    console.log(`Current snapshot: ${snapshot.snapshotNumber}`);
    console.log(`Documents processed: ${snapshot.trainingProgress.totalDocuments}`);
    console.log(`Vocabulary size: ${snapshot.stats.vocabSize}`);
    console.log(`Training phase: ${snapshot.trainingProgress.phase}\n`);

    // Update config from snapshot
    CONFIG.frequencyDim = snapshot.config.frequencyDim;
    CONFIG.windowSize = snapshot.config.windowSize;

    console.log(`Analyzing polysemy for word: "${targetWord}"`);
    console.log(`Comparison (monosemous) word: "${comparisonWord}"\n`);

    // Load corpus data
    console.log('Loading corpus from parquet files...');
    const parquetFiles = fs.readdirSync(CONFIG.parquetDir)
        .filter(f => f.endsWith('.parquet'))
        .sort()
        .map(f => path.join(CONFIG.parquetDir, f));

    if (parquetFiles.length === 0) {
        console.error('No parquet files found');
        process.exit(1);
    }

    // Load limited number of documents
    const documents = [];
    for (const file of parquetFiles.slice(0, 2)) {  // Load first 2 files only
        const docs = loadParquetFile(file);
        documents.push(...docs);
        if (documents.length >= 100) break;  // Limit to 100 docs for analysis
    }
    console.log(`Total documents for analysis: ${documents.length}\n`);

    // Extract contexts
    console.log(`Extracting contexts for "${targetWord}"...`);
    const contexts = extractContexts(targetWord, snapshot, documents);
    console.log(`Found ${contexts.length} contexts\n`);

    if (contexts.length < 10) {
        console.log(`Too few contexts found for meaningful analysis (${contexts.length} < 10)`);
        process.exit(0);
    }

    // Build context spectra
    console.log('Building context spectra...');
    const contextSpectra = contexts.map(ctx =>
        buildContextSpectrum(ctx.contextWords, CONFIG.frequencyDim)
    );

    // Cluster contexts (2 clusters for polysemy)
    console.log('Clustering contexts...');
    const clustering = clusterContexts(contextSpectra, 2);

    if (!clustering) {
        console.log('Failed to cluster contexts');
        process.exit(1);
    }

    // Calculate metrics
    const silhouette = calculateSilhouette(contextSpectra, clustering.assignments);

    // ======================================
    // TEST 1: MULTIPLE DOMINANT PEAKS
    // ======================================
    console.log('\n' + '='.repeat(70));
    console.log('TEST 1: MULTIPLE DOMINANT PEAKS');
    console.log('='.repeat(70) + '\n');

    const targetPeaks = analyzeSpectralPeaks(
        snapshot.vocabulary.find(w => w.word === targetWord)?.spectrum
    );

    const comparisonPeaks = analyzeSpectralPeaks(
        snapshot.vocabulary.find(w => w.word === comparisonWord)?.spectrum
    );

    console.log(`${targetWord.toUpperCase()}:`);
    if (targetPeaks.peaks.length > 0) {
        console.log(`  Dominant peaks (amplitude > ${CONFIG.multiPeakThreshold}):`);
        targetPeaks.peaks.forEach((peak, idx) => {
            console.log(`    ${idx + 1}. Freq ${peak.frequency}: amplitude ${peak.amplitude.toFixed(4)}`);
        });
        console.log(`  Total peaks: ${targetPeaks.peakCount}`);
        console.log(`  Potentially polysemous: ${targetPeaks.isPotentiallyPolysemous ? 'YES ✓' : 'NO'}`);
    } else {
        console.log('  No spectrum data available');
    }

    console.log(`\n${comparisonWord.toUpperCase()} (comparison):`);
    if (comparisonPeaks.peaks.length > 0) {
        console.log(`  Dominant peaks (amplitude > ${CONFIG.multiPeakThreshold}):`);
        comparisonPeaks.peaks.forEach((peak, idx) => {
            console.log(`    ${idx + 1}. Freq ${peak.frequency}: amplitude ${peak.amplitude.toFixed(4)}`);
        });
        console.log(`  Total peaks: ${comparisonPeaks.peakCount}`);
        console.log(`  Potentially polysemous: ${comparisonPeaks.isPotentiallyPolysemous ? 'YES' : 'NO ✓'}`);
    } else {
        console.log('  No spectrum data available');
    }

    console.log('\nInterpretation:');
    console.log('  Polysemous words should have 2+ dominant peaks');
    console.log('  Monosemous words should have 1 dominant peak');

    // ======================================
    // TEST 2: DIVERGENCE OVER SNAPSHOTS
    // ======================================
    if (allSnapshots.length > 1) {
        console.log('\n' + '='.repeat(70));
        console.log('TEST 2: DIVERGENCE OVER TRAINING SNAPSHOTS');
        console.log('='.repeat(70) + '\n');

        const targetEvolution = trackFrequencyEvolution(targetWord, allSnapshots);
        const comparisonEvolution = trackFrequencyEvolution(comparisonWord, allSnapshots);

        console.log(`${targetWord.toUpperCase()} frequency evolution:`);
        console.log('  Snapshot | Docs    | Active Freqs | Top 3 Amplitudes');
        console.log('  ' + '-'.repeat(60));

        targetEvolution.forEach(evo => {
            if (evo.spectrum) {
                const topAmps = [...evo.spectrum.amplitudes]
                    .sort((a, b) => b - a)
                    .slice(0, 3)
                    .map(a => a.toFixed(3))
                    .join(', ');
                console.log(`  ${String(evo.snapshotNumber).padStart(8)} | ${String(evo.documents).padStart(7)} | ${String(evo.spectrum.frequencies.length).padStart(12)} | ${topAmps}`);
            }
        });

        console.log(`\n${comparisonWord.toUpperCase()} frequency evolution:`);
        console.log('  Snapshot | Docs    | Active Freqs | Top 3 Amplitudes');
        console.log('  ' + '-'.repeat(60));

        comparisonEvolution.forEach(evo => {
            if (evo.spectrum) {
                const topAmps = [...evo.spectrum.amplitudes]
                    .sort((a, b) => b - a)
                    .slice(0, 3)
                    .map(a => a.toFixed(3))
                    .join(', ');
                console.log(`  ${String(evo.snapshotNumber).padStart(8)} | ${String(evo.documents).padStart(7)} | ${String(evo.spectrum.frequencies.length).padStart(12)} | ${topAmps}`);
            }
        });

        console.log('\nInterpretation:');
        console.log('  Polysemous words: Multiple peaks grow at different rates');
        console.log('  Monosemous words: Single dominant peak emerges consistently');
    }

    // ======================================
    // TEST 3: CONTEXT CLUSTERING
    // ======================================
    console.log('\n' + '='.repeat(70));
    console.log('TEST 3: CONTEXT CLUSTERING');
    console.log('='.repeat(70) + '\n');

    const cluster0Count = clustering.assignments.filter(a => a === 0).length;
    const cluster1Count = clustering.assignments.filter(a => a === 1).length;

    console.log(`Cluster 0: ${cluster0Count} contexts`);
    console.log(`Cluster 1: ${cluster1Count} contexts`);
    console.log(`Silhouette score: ${silhouette.toFixed(4)}`);
    console.log(`  (>0.3 = good separation, 0.0-0.3 = weak, <0.0 = poor)\n`);

    // Show example contexts from each cluster
    console.log('Example contexts from each cluster:\n');

    for (let cluster = 0; cluster < 2; cluster++) {
        console.log(`Cluster ${cluster} examples:`);
        const clusterContexts = contexts.filter((_, idx) =>
            clustering.assignments[idx] === cluster
        );

        clusterContexts.slice(0, 5).forEach((ctx, idx) => {
            console.log(`  ${idx + 1}. ${ctx.tokens.join(' ')}`);
        });
        console.log('');
    }

    console.log('Interpretation:');
    console.log('  Contexts should naturally separate into semantic groups');
    console.log('  Good silhouette score indicates distinct usage patterns');

    // ======================================
    // TEST 4: MUTUAL REINFORCEMENT
    // ======================================
    console.log('\n' + '='.repeat(70));
    console.log('TEST 4: MUTUAL REINFORCEMENT BETWEEN SENSE CLUSTERS');
    console.log('='.repeat(70) + '\n');

    const reinforcement = analyzeMutualReinforcement(targetWord, snapshot, contexts, clustering);

    if (reinforcement) {
        console.log('Top frequencies for each cluster centroid:\n');

        for (let c = 0; c < reinforcement.clusterFrequencies.length; c++) {
            console.log(`Cluster ${c}:`);
            reinforcement.clusterFrequencies[c].forEach((f, idx) => {
                console.log(`  ${idx + 1}. Freq ${f.frequency}: amplitude ${f.amplitude.toFixed(4)}`);
            });
            console.log('');
        }

        console.log(`Frequency overlap between clusters: ${(reinforcement.overlapRatio * 100).toFixed(1)}%`);
        console.log(`Overlapping frequencies: [${reinforcement.overlap.join(', ')}]`);
        console.log(`Clean separation: ${reinforcement.isSeparated ? 'YES ✓' : 'NO'}`);

        console.log('\nInterpretation:');
        console.log('  Low overlap (<30%) indicates distinct sense frequencies');
        console.log('  High overlap suggests senses share semantic components');
    } else {
        console.log('Unable to perform mutual reinforcement analysis');
    }

    // ======================================
    // TEST 5: CONTEXT SUBSTITUTION
    // ======================================
    console.log('\n' + '='.repeat(70));
    console.log('TEST 5: SPECTRUM STABILITY UNDER CONTEXT SUBSTITUTION');
    console.log('='.repeat(70) + '\n');

    const substitution = testContextSubstitution(targetWord, snapshot, contexts, clustering);

    if (substitution) {
        console.log('Average word-to-context similarity by cluster:\n');

        for (const sim of substitution.clusterSimilarities) {
            console.log(`Cluster ${sim.cluster}: ${sim.avgSimilarity.toFixed(4)} (${sim.count} contexts)`);
        }

        console.log(`\nDifferentiated by context: ${substitution.isDifferentiated ? 'YES ✓' : 'NO'}`);

        console.log('\nInterpretation:');
        console.log('  Different similarity scores per cluster indicate sense disambiguation');
        console.log('  Word spectrum should resonate differently with each sense context');
    } else {
        console.log('Unable to perform context substitution analysis');
    }

    // ======================================
    // FINAL SUMMARY
    // ======================================
    console.log('\n' + '█'.repeat(70));
    console.log('POLYSEMY DETECTION SUMMARY');
    console.log('█'.repeat(70) + '\n');

    let polysemyScore = 0;
    let totalTests = 0;

    console.log(`Word: "${targetWord}"\n`);
    console.log('Test Results:');

    // Test 1
    totalTests++;
    if (targetPeaks.isPotentiallyPolysemous) {
        console.log(`  ✓ Test 1 (Multiple Peaks): PASS - ${targetPeaks.peakCount} dominant peaks`);
        polysemyScore++;
    } else {
        console.log(`  ✗ Test 1 (Multiple Peaks): FAIL - ${targetPeaks.peakCount} dominant peak(s)`);
    }

    // Test 2 (only if multiple snapshots)
    if (allSnapshots.length > 1) {
        totalTests++;
        // Check if we have divergence (simplified check)
        console.log(`  ? Test 2 (Divergence): See temporal evolution above`);
    }

    // Test 3
    totalTests++;
    if (silhouette > 0.3) {
        console.log(`  ✓ Test 3 (Clustering): PASS - Silhouette ${silhouette.toFixed(4)}`);
        polysemyScore++;
    } else if (silhouette > 0.0) {
        console.log(`  ~ Test 3 (Clustering): WEAK - Silhouette ${silhouette.toFixed(4)}`);
        polysemyScore += 0.5;
    } else {
        console.log(`  ✗ Test 3 (Clustering): FAIL - Silhouette ${silhouette.toFixed(4)}`);
    }

    // Test 4
    if (reinforcement) {
        totalTests++;
        if (reinforcement.isSeparated) {
            console.log(`  ✓ Test 4 (Mutual Reinforcement): PASS - ${(reinforcement.overlapRatio * 100).toFixed(1)}% overlap`);
            polysemyScore++;
        } else {
            console.log(`  ✗ Test 4 (Mutual Reinforcement): FAIL - ${(reinforcement.overlapRatio * 100).toFixed(1)}% overlap`);
        }
    }

    // Test 5
    if (substitution) {
        totalTests++;
        if (substitution.isDifferentiated) {
            console.log(`  ✓ Test 5 (Context Substitution): PASS - Contexts differentiated`);
            polysemyScore++;
        } else {
            console.log(`  ✗ Test 5 (Context Substitution): FAIL - No differentiation`);
        }
    }

    console.log(`\nPolysemy Score: ${polysemyScore}/${totalTests}`);
    console.log(`Confidence: ${(polysemyScore / totalTests * 100).toFixed(1)}%\n`);

    if (polysemyScore / totalTests >= 0.6) {
        console.log('✓ POLYSEMY DETECTED: Model successfully learned multiple senses');
    } else if (polysemyScore / totalTests >= 0.3) {
        console.log('~ WEAK POLYSEMY: Some evidence of sense separation');
    } else {
        console.log('✗ NO POLYSEMY: Word appears monosemous or needs more training');
    }

    console.log('\n' + '█'.repeat(70) + '\n');
}

// Run analysis
analyzePolysemy().catch(console.error);
