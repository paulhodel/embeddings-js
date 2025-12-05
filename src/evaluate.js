/**
 * Evaluation Script - Semantic Quality Testing
 * Tests word embeddings for semantic relationships, similarity, and polysemy
 */

import fs from 'fs';

const CONFIG = {
    modelFile: './data/model2_final_gpu.ndjson',  // Or checkpoint file
    embeddingDim: 64,
};

async function main() {
    console.log('\n' + '='.repeat(60));
    console.log('SEMANTIC EVALUATION');
    console.log('='.repeat(60));

    // ============================================
    // LOAD MODEL
    // ============================================
    console.log(`\nLoading model from: ${CONFIG.modelFile}`);

    const lines = fs.readFileSync(CONFIG.modelFile, 'utf8').split('\n');
    const embeddings = [];
    const wordToId = new Map();
    const idToWord = [];

    const embDim = CONFIG.embeddingDim;
    let lineCount = 0;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line.length === 0) continue;

        const parts = line.split('\t');
        const word = parts[0];
        const id = parseInt(parts[1]);
        const vectorStr = parts[2];

        const vectorParts = vectorStr.split(' ');
        const vector = new Array(embDim);
        for (let j = 0; j < embDim; j++) {
            vector[j] = parseFloat(vectorParts[j]);
        }

        embeddings[id] = vector;
        wordToId.set(word, id);
        idToWord[id] = word;
        lineCount++;
    }

    console.log(`Loaded ${lineCount} word embeddings`);

    const vocabSize = embeddings.length;

    // ============================================
    // HELPER FUNCTIONS
    // ============================================

    /**
     * Compute cosine similarity between two vectors
     */
    function cosineSimilarity(vec1, vec2) {
        let dot = 0;
        let mag1 = 0;
        let mag2 = 0;

        for (let i = 0; i < embDim; i++) {
            dot += vec1[i] * vec2[i];
            mag1 += vec1[i] * vec1[i];
            mag2 += vec2[i] * vec2[i];
        }

        return dot / (Math.sqrt(mag1) * Math.sqrt(mag2));
    }

    /**
     * Find K nearest neighbors for a word
     */
    function findNearest(word, k = 10) {
        const wordId = wordToId.get(word);
        if (wordId === undefined) return null;

        const wordVec = embeddings[wordId];
        const similarities = [];

        for (let i = 0; i < vocabSize; i++) {
            if (i === wordId) continue;

            const sim = cosineSimilarity(wordVec, embeddings[i]);
            similarities.push({ word: idToWord[i], similarity: sim });
        }

        similarities.sort((a, b) => b.similarity - a.similarity);
        return similarities.slice(0, k);
    }

    /**
     * Test word analogy: word1 - word2 + word3 ≈ ?
     */
    function testAnalogy(word1, word2, word3) {
        const id1 = wordToId.get(word1);
        const id2 = wordToId.get(word2);
        const id3 = wordToId.get(word3);

        if (id1 === undefined || id2 === undefined || id3 === undefined) {
            return null;
        }

        // Compute result vector
        const result = new Array(embDim);
        for (let d = 0; d < embDim; d++) {
            result[d] = embeddings[id3][d] + (embeddings[id2][d] - embeddings[id1][d]);
        }

        // Find nearest
        let bestId = -1;
        let bestSim = -Infinity;

        for (let candidateId = 0; candidateId < vocabSize; candidateId++) {
            if (candidateId === id1 || candidateId === id2 || candidateId === id3) continue;

            const sim = cosineSimilarity(result, embeddings[candidateId]);
            if (sim > bestSim) {
                bestSim = sim;
                bestId = candidateId;
            }
        }

        return { word: idToWord[bestId], similarity: bestSim };
    }

    // ============================================
    // TEST 1: WORD SIMILARITY
    // ============================================
    console.log('\n' + '='.repeat(60));
    console.log('TEST 1: WORD SIMILARITY');
    console.log('='.repeat(60));

    const similarityTests = [
        'king', 'queen', 'man', 'woman',
        'good', 'bad', 'hot', 'cold',
        'big', 'small', 'fast', 'slow',
        'water', 'fire', 'earth', 'air',
        'cat', 'dog', 'bird', 'fish',
        'computer', 'technology', 'science', 'math'
    ];

    console.log('\nFinding nearest neighbors for test words:\n');

    for (let i = 0; i < similarityTests.length; i++) {
        const word = similarityTests[i];
        const nearest = findNearest(word, 5);

        if (nearest) {
            console.log(`"${word}":`);
            for (let j = 0; j < nearest.length; j++) {
                const sim = nearest[j].similarity.toFixed(4);
                console.log(`  ${j + 1}. ${nearest[j].word} (${sim})`);
            }
            console.log('');
        } else {
            console.log(`"${word}": Not in vocabulary\n`);
        }
    }

    // ============================================
    // TEST 2: WORD ANALOGIES
    // ============================================
    console.log('='.repeat(60));
    console.log('TEST 2: WORD ANALOGIES');
    console.log('='.repeat(60));

    const analogyTests = [
        { word1: 'king', word2: 'queen', word3: 'man', expect: 'woman' },
        { word1: 'brother', word2: 'sister', word3: 'father', expect: 'mother' },
        { word1: 'big', word2: 'bigger', word3: 'small', expect: 'smaller' },
        { word1: 'good', word2: 'better', word3: 'bad', expect: 'worse' },
        { word1: 'walk', word2: 'walking', word3: 'run', expect: 'running' },
        { word1: 'france', word2: 'paris', word3: 'england', expect: 'london' },
        { word1: 'japan', word2: 'tokyo', word3: 'china', expect: 'beijing' },
        { word1: 'write', word2: 'writing', word3: 'read', expect: 'reading' },
    ];

    console.log('\nTesting analogies:\n');

    let correct = 0;
    let tested = 0;

    for (let i = 0; i < analogyTests.length; i++) {
        const test = analogyTests[i];
        const result = testAnalogy(test.word1, test.word2, test.word3);

        if (result) {
            tested++;
            const isCorrect = result.word === test.expect;
            if (isCorrect) correct++;

            const mark = isCorrect ? '✓' : '✗';
            console.log(`${test.word1} - ${test.word2} + ${test.word3} = ${result.word} (expect: ${test.expect}) ${mark}`);
        } else {
            console.log(`${test.word1} - ${test.word2} + ${test.word3} = ? (words not in vocab)`);
        }
    }

    if (tested > 0) {
        const accuracy = (correct / tested * 100).toFixed(1);
        console.log(`\nAnalogy accuracy: ${correct}/${tested} (${accuracy}%)`);
    }

    // ============================================
    // TEST 3: SEMANTIC CLUSTERING
    // Check if semantically similar words cluster together
    // ============================================
    console.log('\n' + '='.repeat(60));
    console.log('TEST 3: SEMANTIC CLUSTERING');
    console.log('='.repeat(60));

    const clusters = [
        { name: 'Animals', words: ['cat', 'dog', 'bird', 'fish', 'lion', 'tiger'] },
        { name: 'Colors', words: ['red', 'blue', 'green', 'yellow', 'black', 'white'] },
        { name: 'Numbers', words: ['one', 'two', 'three', 'four', 'five', 'six'] },
        { name: 'Family', words: ['mother', 'father', 'sister', 'brother', 'child', 'parent'] },
        { name: 'Emotions', words: ['happy', 'sad', 'angry', 'joy', 'fear', 'love'] }
    ];

    console.log('\nAverage intra-cluster similarity:\n');

    for (let c = 0; c < clusters.length; c++) {
        const cluster = clusters[c];
        const words = cluster.words;

        // Get word IDs that exist in vocab
        const validIds = [];
        for (let i = 0; i < words.length; i++) {
            const id = wordToId.get(words[i]);
            if (id !== undefined) validIds.push(id);
        }

        if (validIds.length < 2) {
            console.log(`${cluster.name}: Insufficient words in vocabulary (${validIds.length}/${words.length})`);
            continue;
        }

        // Compute average pairwise similarity
        let totalSim = 0;
        let pairs = 0;

        for (let i = 0; i < validIds.length; i++) {
            for (let j = i + 1; j < validIds.length; j++) {
                const sim = cosineSimilarity(embeddings[validIds[i]], embeddings[validIds[j]]);
                totalSim += sim;
                pairs++;
            }
        }

        const avgSim = totalSim / pairs;
        console.log(`${cluster.name}: ${avgSim.toFixed(4)} (${validIds.length}/${words.length} words)`);
    }

    // ============================================
    // TEST 4: POLYSEMY DETECTION
    // Words with multiple meanings should have moderate similarity to different semantic fields
    // ============================================
    console.log('\n' + '='.repeat(60));
    console.log('TEST 4: POLYSEMY ANALYSIS');
    console.log('='.repeat(60));

    const polysemousWords = [
        { word: 'bank', meanings: ['river bank', 'financial bank'] },
        { word: 'bat', meanings: ['baseball bat', 'flying bat'] },
        { word: 'light', meanings: ['not heavy', 'brightness'] },
        { word: 'spring', meanings: ['season', 'coil'] },
        { word: 'match', meanings: ['game', 'fire starter'] }
    ];

    console.log('\nAnalyzing polysemous words (nearest neighbors):\n');

    for (let i = 0; i < polysemousWords.length; i++) {
        const item = polysemousWords[i];
        const nearest = findNearest(item.word, 10);

        if (nearest) {
            console.log(`"${item.word}" (meanings: ${item.meanings.join(', ')}):`);
            for (let j = 0; j < Math.min(5, nearest.length); j++) {
                const sim = nearest[j].similarity.toFixed(4);
                console.log(`  ${j + 1}. ${nearest[j].word} (${sim})`);
            }
            console.log('');
        } else {
            console.log(`"${item.word}": Not in vocabulary\n`);
        }
    }

    // ============================================
    // TEST 5: ANTONYM DETECTION
    // Antonyms should have lower similarity than synonyms
    // ============================================
    console.log('='.repeat(60));
    console.log('TEST 5: ANTONYM vs SYNONYM SIMILARITY');
    console.log('='.repeat(60));

    const antonymPairs = [
        { word: 'good', synonym: 'great', antonym: 'bad' },
        { word: 'hot', synonym: 'warm', antonym: 'cold' },
        { word: 'big', synonym: 'large', antonym: 'small' },
        { word: 'fast', synonym: 'quick', antonym: 'slow' },
        { word: 'happy', synonym: 'joyful', antonym: 'sad' }
    ];

    console.log('\nComparing synonym vs antonym similarity:\n');

    for (let i = 0; i < antonymPairs.length; i++) {
        const pair = antonymPairs[i];
        const wordId = wordToId.get(pair.word);
        const synId = wordToId.get(pair.synonym);
        const antId = wordToId.get(pair.antonym);

        if (wordId !== undefined && synId !== undefined && antId !== undefined) {
            const synSim = cosineSimilarity(embeddings[wordId], embeddings[synId]);
            const antSim = cosineSimilarity(embeddings[wordId], embeddings[antId]);

            console.log(`"${pair.word}":`);
            console.log(`  Synonym "${pair.synonym}": ${synSim.toFixed(4)}`);
            console.log(`  Antonym "${pair.antonym}": ${antSim.toFixed(4)}`);
            console.log(`  Difference: ${(synSim - antSim).toFixed(4)} ${synSim > antSim ? '✓' : '✗'}`);
            console.log('');
        } else {
            console.log(`"${pair.word}": Words not in vocabulary\n`);
        }
    }

    // ============================================
    // COMPLETE
    // ============================================
    console.log('='.repeat(60));
    console.log('EVALUATION COMPLETE');
    console.log('='.repeat(60));
    console.log('');
}

main().catch(error => {
    console.error('\n❌ Evaluation failed:', error.message);
    console.error(error.stack);
    process.exit(1);
});
