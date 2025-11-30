/**
 * Checkpoint Comparison Script
 * Tracks semantic quality metrics over time across multiple checkpoints
 */

import fs from 'fs';
import path from 'path';

const CONFIG = {
    checkpointDir: './data/checkpoints2',
    outputFile: './data/metrics_over_time.json',
    embeddingDim: 64,
};

async function main() {
    console.log('\n' + '='.repeat(60));
    console.log('CHECKPOINT COMPARISON - METRICS OVER TIME');
    console.log('='.repeat(60));

    // ============================================
    // FIND ALL CHECKPOINTS
    // ============================================
    console.log(`\nScanning checkpoint directory: ${CONFIG.checkpointDir}`);

    let checkpointFiles = fs.existsSync(CONFIG.checkpointDir)
        ? fs.readdirSync(CONFIG.checkpointDir).filter(f => f.startsWith('checkpoint_epoch_'))
        : [];

    if (checkpointFiles.length === 0) {
        console.log('No checkpoints found!');
        process.exit(1);
    }

    // Sort numerically by batch number
    checkpointFiles.sort((a, b) => {
        const batchA = parseInt(a.match(/batch_(\d+)/)[1]);
        const batchB = parseInt(b.match(/batch_(\d+)/)[1]);
        return batchA - batchB;
    });
    console.log(`Found ${checkpointFiles.length} checkpoints`);

    // Sample every 10th checkpoint if there are more than 10
    if (checkpointFiles.length > 10) {
        const step = Math.floor(checkpointFiles.length / 10);
        const selectedCheckpoints = [];
        for (let i = 0; i < checkpointFiles.length; i += step) {
            selectedCheckpoints.push(checkpointFiles[i]);
        }
        // Always include the last checkpoint
        if (selectedCheckpoints[selectedCheckpoints.length - 1] !== checkpointFiles[checkpointFiles.length - 1]) {
            selectedCheckpoints.push(checkpointFiles[checkpointFiles.length - 1]);
        }
        console.log(`Sampling every ${step} checkpoints (selected ${selectedCheckpoints.length})`);
        checkpointFiles = selectedCheckpoints;
    }

    // ============================================
    // DEFINE TEST SUITE
    // ============================================

    const analogyTests = [
        { word1: 'big', word2: 'bigger', word3: 'small', expect: 'smaller' },
        { word1: 'walk', word2: 'walking', word3: 'talk', expect: 'talking' },
        { word1: 'brother', word2: 'sister', word3: 'father', expect: 'mother' },
        { word1: 'france', word2: 'paris', word3: 'england', expect: 'london' },
        { word1: 'good', word2: 'better', word3: 'bad', expect: 'worse' },
        { word1: 'king', word2: 'queen', word3: 'man', expect: 'woman' },
    ];

    const similarityPairs = [
        { word1: 'cat', word2: 'dog' },
        { word1: 'king', word2: 'queen' },
        { word1: 'man', word2: 'woman' },
        { word1: 'good', word2: 'bad' },
        { word1: 'hot', word2: 'cold' },
    ];

    const synonymAntonymTests = [
        { word: 'good', synonym: 'great', antonym: 'bad' },
        { word: 'big', synonym: 'large', antonym: 'small' },
        { word: 'fast', synonym: 'quick', antonym: 'slow' },
    ];

    const categoryTests = [
        // Animals - should cluster together
        { name: 'Animals', words: ['cat', 'dog', 'bird', 'fish', 'lion'] },
        // Numbers - should cluster together
        { name: 'Numbers', words: ['one', 'two', 'three', 'four', 'five'] },
        // Family - should cluster together
        { name: 'Family', words: ['mother', 'father', 'sister', 'brother', 'child'] },
        // Colors - should cluster together
        { name: 'Colors', words: ['red', 'blue', 'green', 'yellow', 'black'] },
    ];

    const distinctionTests = [
        // These pairs should be LESS similar (different categories)
        { word1: 'cat', word2: 'car', expectLow: true },  // animal vs vehicle
        { word1: 'happy', word2: 'table', expectLow: true },  // emotion vs object
        { word1: 'run', word2: 'mountain', expectLow: true },  // verb vs noun
        { word1: 'king', word2: 'three', expectLow: true },  // title vs number
    ];

    // ============================================
    // HELPER FUNCTIONS
    // ============================================

    function cosineSimilarity(vec1, vec2, dim) {
        let dot = 0;
        let mag1 = 0;
        let mag2 = 0;

        for (let i = 0; i < dim; i++) {
            dot += vec1[i] * vec2[i];
            mag1 += vec1[i] * vec1[i];
            mag2 += vec2[i] * vec2[i];
        }

        return dot / (Math.sqrt(mag1) * Math.sqrt(mag2));
    }

    function testAnalogy(embeddings, wordToId, idToWord, word1, word2, word3, expect, dim) {
        const id1 = wordToId.get(word1);
        const id2 = wordToId.get(word2);
        const id3 = wordToId.get(word3);
        const expectId = wordToId.get(expect);

        if (id1 === undefined || id2 === undefined || id3 === undefined || expectId === undefined) {
            return null;
        }

        const vocabSize = embeddings.length;

        // Compute result vector
        const result = new Array(dim);
        for (let d = 0; d < dim; d++) {
            result[d] = embeddings[id3][d] + (embeddings[id2][d] - embeddings[id1][d]);
        }

        // Find nearest
        let bestId = -1;
        let bestSim = -Infinity;

        for (let candidateId = 0; candidateId < vocabSize; candidateId++) {
            if (candidateId === id1 || candidateId === id2 || candidateId === id3) continue;

            const sim = cosineSimilarity(result, embeddings[candidateId], dim);
            if (sim > bestSim) {
                bestSim = sim;
                bestId = candidateId;
            }
        }

        return { correct: idToWord[bestId] === expect, predicted: idToWord[bestId] };
    }

    // ============================================
    // PROCESS EACH CHECKPOINT
    // ============================================
    const results = [];

    for (let cpIdx = 0; cpIdx < checkpointFiles.length; cpIdx++) {
        const filename = checkpointFiles[cpIdx];
        const filepath = path.join(CONFIG.checkpointDir, filename);

        console.log(`\n[${cpIdx + 1}/${checkpointFiles.length}] Processing ${filename}...`);

        // Load checkpoint
        const checkpointData = JSON.parse(fs.readFileSync(filepath, 'utf8'));
        const embeddings = checkpointData.embeddings;
        const wordToId = new Map(checkpointData.wordToId);
        const idToWord = checkpointData.idToWord;
        const epoch = checkpointData.epoch;
        const batch = checkpointData.batch;
        const avgLoss = checkpointData.avgLoss;

        const embDim = CONFIG.embeddingDim;

        // Compute metrics
        const metrics = {
            checkpoint: filename,
            epoch: epoch,
            batch: batch,
            avgLoss: avgLoss,
            timestamp: checkpointData.timestamp,
        };

        // TEST 1: Analogy Accuracy
        let analogyCorrect = 0;
        let analogyTested = 0;

        for (let i = 0; i < analogyTests.length; i++) {
            const test = analogyTests[i];
            const result = testAnalogy(
                embeddings, wordToId, idToWord,
                test.word1, test.word2, test.word3, test.expect, embDim
            );

            if (result) {
                analogyTested++;
                if (result.correct) analogyCorrect++;
            }
        }

        metrics.analogyAccuracy = analogyTested > 0 ? (analogyCorrect / analogyTested) : 0;
        metrics.analogyTested = analogyTested;
        metrics.analogyCorrect = analogyCorrect;

        // TEST 2: Average Similarity of Related Pairs
        let totalSimilarity = 0;
        let pairsTested = 0;

        for (let i = 0; i < similarityPairs.length; i++) {
            const pair = similarityPairs[i];
            const id1 = wordToId.get(pair.word1);
            const id2 = wordToId.get(pair.word2);

            if (id1 !== undefined && id2 !== undefined) {
                const sim = cosineSimilarity(embeddings[id1], embeddings[id2], embDim);
                totalSimilarity += sim;
                pairsTested++;
            }
        }

        metrics.avgPairSimilarity = pairsTested > 0 ? (totalSimilarity / pairsTested) : 0;
        metrics.pairsTested = pairsTested;

        // TEST 3: Synonym vs Antonym Discrimination
        let synAntCorrect = 0;
        let synAntTested = 0;

        for (let i = 0; i < synonymAntonymTests.length; i++) {
            const test = synonymAntonymTests[i];
            const wordId = wordToId.get(test.word);
            const synId = wordToId.get(test.synonym);
            const antId = wordToId.get(test.antonym);

            if (wordId !== undefined && synId !== undefined && antId !== undefined) {
                const synSim = cosineSimilarity(embeddings[wordId], embeddings[synId], embDim);
                const antSim = cosineSimilarity(embeddings[wordId], embeddings[antId], embDim);

                synAntTested++;
                if (synSim > antSim) synAntCorrect++;
            }
        }

        metrics.synAntAccuracy = synAntTested > 0 ? (synAntCorrect / synAntTested) : 0;
        metrics.synAntTested = synAntTested;
        metrics.synAntCorrect = synAntCorrect;

        // TEST 4: Category Clustering (intra-category similarity)
        const categoryScores = [];
        for (let c = 0; c < categoryTests.length; c++) {
            const category = categoryTests[c];
            const validIds = [];

            for (let w = 0; w < category.words.length; w++) {
                const id = wordToId.get(category.words[w]);
                if (id !== undefined) validIds.push(id);
            }

            if (validIds.length >= 2) {
                let totalSim = 0;
                let pairs = 0;

                for (let i = 0; i < validIds.length; i++) {
                    for (let j = i + 1; j < validIds.length; j++) {
                        const sim = cosineSimilarity(embeddings[validIds[i]], embeddings[validIds[j]], embDim);
                        totalSim += sim;
                        pairs++;
                    }
                }

                const avgSim = totalSim / pairs;
                categoryScores.push(avgSim);
            }
        }

        metrics.avgCategorySimilarity = categoryScores.length > 0
            ? (categoryScores.reduce((a, b) => a + b, 0) / categoryScores.length)
            : 0;
        metrics.categoryTestsRun = categoryScores.length;

        // TEST 5: Cross-category Distinction (should be lower similarity)
        let distinctionCorrect = 0;
        let distinctionTested = 0;
        let avgCrossCategorySim = 0;

        for (let i = 0; i < distinctionTests.length; i++) {
            const test = distinctionTests[i];
            const id1 = wordToId.get(test.word1);
            const id2 = wordToId.get(test.word2);

            if (id1 !== undefined && id2 !== undefined) {
                const sim = cosineSimilarity(embeddings[id1], embeddings[id2], embDim);
                avgCrossCategorySim += sim;
                distinctionTested++;

                // For cross-category pairs, similarity should be lower than avg category similarity
                if (metrics.avgCategorySimilarity > 0 && sim < metrics.avgCategorySimilarity) {
                    distinctionCorrect++;
                }
            }
        }

        metrics.avgCrossCategorySim = distinctionTested > 0 ? (avgCrossCategorySim / distinctionTested) : 0;
        metrics.distinctionAccuracy = distinctionTested > 0 ? (distinctionCorrect / distinctionTested) : 0;
        metrics.distinctionTested = distinctionTested;

        // TEST 6: Semantic Separation Score (category sim should be > cross-category sim)
        metrics.separationScore = metrics.avgCategorySimilarity - metrics.avgCrossCategorySim;

        results.push(metrics);

        console.log(`  Loss: ${avgLoss.toFixed(6)}`);
        console.log(`  Analogy: ${analogyCorrect}/${analogyTested} (${(metrics.analogyAccuracy * 100).toFixed(1)}%)`);
        console.log(`  Pair Sim: ${metrics.avgPairSimilarity.toFixed(4)}`);
        console.log(`  Syn>Ant: ${synAntCorrect}/${synAntTested} (${(metrics.synAntAccuracy * 100).toFixed(1)}%)`);
        console.log(`  Category Cluster: ${metrics.avgCategorySimilarity.toFixed(4)}`);
        console.log(`  Cross-Category: ${metrics.avgCrossCategorySim.toFixed(4)}`);
        console.log(`  Separation: ${metrics.separationScore.toFixed(4)}`);
    }

    // ============================================
    // SAVE RESULTS
    // ============================================
    console.log(`\n\nSaving results to: ${CONFIG.outputFile}`);
    fs.writeFileSync(CONFIG.outputFile, JSON.stringify(results, null, 2));

    // ============================================
    // DISPLAY SUMMARY TABLE
    // ============================================
    console.log('\n' + '='.repeat(100));
    console.log('METRICS OVER TIME - SUMMARY');
    console.log('='.repeat(100));
    console.log('');
    console.log('Batch       | Loss      | Analogy | Category | CrossCat | Separation | Checkpoint');
    console.log('-'.repeat(100));

    for (let i = 0; i < results.length; i++) {
        const r = results[i];
        const batchStr = r.batch.toString().padEnd(11);
        const lossStr = r.avgLoss.toFixed(6).padEnd(9);
        const analogyStr = `${(r.analogyAccuracy * 100).toFixed(0)}%`.padEnd(7);
        const categoryStr = r.avgCategorySimilarity.toFixed(4).padEnd(8);
        const crossCatStr = r.avgCrossCategorySim.toFixed(4).padEnd(8);
        const sepStr = r.separationScore.toFixed(4).padEnd(10);

        console.log(`${batchStr} | ${lossStr} | ${analogyStr} | ${categoryStr} | ${crossCatStr} | ${sepStr} | ${r.checkpoint}`);
    }

    console.log('='.repeat(100));

    // ============================================
    // SHOW TRENDS
    // ============================================
    if (results.length > 1) {
        console.log('\n' + '='.repeat(60));
        console.log('TRENDS (First → Last)');
        console.log('='.repeat(60));

        const first = results[0];
        const last = results[results.length - 1];

        const lossDelta = last.avgLoss - first.avgLoss;
        const analogyDelta = (last.analogyAccuracy - first.analogyAccuracy) * 100;
        const categoryDelta = last.avgCategorySimilarity - first.avgCategorySimilarity;
        const crossCatDelta = last.avgCrossCategorySim - first.avgCrossCategorySim;
        const sepDelta = last.separationScore - first.separationScore;

        console.log(`Loss:                    ${first.avgLoss.toFixed(6)} → ${last.avgLoss.toFixed(6)} (${lossDelta > 0 ? '+' : ''}${lossDelta.toFixed(6)})`);
        console.log(`Analogy Accuracy:        ${(first.analogyAccuracy * 100).toFixed(1)}% → ${(last.analogyAccuracy * 100).toFixed(1)}% (${analogyDelta > 0 ? '+' : ''}${analogyDelta.toFixed(1)}%)`);
        console.log(`Category Similarity:     ${first.avgCategorySimilarity.toFixed(4)} → ${last.avgCategorySimilarity.toFixed(4)} (${categoryDelta > 0 ? '+' : ''}${categoryDelta.toFixed(4)})`);
        console.log(`Cross-Category Sim:      ${first.avgCrossCategorySim.toFixed(4)} → ${last.avgCrossCategorySim.toFixed(4)} (${crossCatDelta > 0 ? '+' : ''}${crossCatDelta.toFixed(4)})`);
        console.log(`Separation Score:        ${first.separationScore.toFixed(4)} → ${last.separationScore.toFixed(4)} (${sepDelta > 0 ? '+' : ''}${sepDelta.toFixed(4)})`);
    }

    console.log('\n' + '='.repeat(60));
    console.log('COMPARISON COMPLETE');
    console.log('='.repeat(60));
    console.log('');
}

main().catch(error => {
    console.error('\n❌ Comparison failed:', error.message);
    console.error(error.stack);
    process.exit(1);
});
