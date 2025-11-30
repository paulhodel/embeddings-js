/**
 * Find Similar Words - Quick lookup script
 * Usage: node src/find_similar.js <word> [top_k] [checkpoint]
 * Examples:
 *   node src/find_similar.js king 10
 *   node src/find_similar.js king 10 checkpoint_epoch_0_batch_500000.json
 */

import fs from 'fs';
import path from 'path';

const args = process.argv.slice(2);
if (args.length === 0) {
    console.log('Usage: node src/find_similar.js <word> [top_k] [checkpoint]');
    console.log('Examples:');
    console.log('  node src/find_similar.js king 10');
    console.log('  node src/find_similar.js king 10 checkpoint_epoch_0_batch_500000.json');
    process.exit(1);
}

const queryWord = args[0].toLowerCase();
const topK = args[1] ? parseInt(args[1]) : 10;
const checkpointFile = args[2];

let embeddings, wordToId, idToWord;

// Load from checkpoint or final model
if (checkpointFile) {
    const checkpointPath = path.join('./data/checkpoints', checkpointFile);
    console.log(`Loading checkpoint: ${checkpointPath}`);

    const checkpoint = JSON.parse(fs.readFileSync(checkpointPath, 'utf8'));
    embeddings = checkpoint.embeddings;
    wordToId = new Map(checkpoint.wordToId);
    idToWord = checkpoint.idToWord;

    console.log(`Loaded checkpoint (epoch ${checkpoint.epoch}, batch ${checkpoint.batch})`);
} else {
    const modelFile = './data/model_final.ndjson';
    console.log(`Loading model from: ${modelFile}`);

    const lines = fs.readFileSync(modelFile, 'utf8').split('\n');
    embeddings = [];
    wordToId = new Map();
    idToWord = [];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line.length === 0) continue;

        const parts = line.split('\t');
        const word = parts[0];
        const id = parseInt(parts[1]);
        const vectorStr = parts[2];

        const vectorParts = vectorStr.split(' ');
        const vector = vectorParts.map(v => parseFloat(v));

        embeddings[id] = vector;
        wordToId.set(word, id);
        idToWord[id] = word;
    }
}

console.log(`Loaded ${embeddings.length} word embeddings\n`);

// Check if word exists
if (!wordToId.has(queryWord)) {
    console.log(`Word "${queryWord}" not found in vocabulary`);
    process.exit(1);
}

// Cosine similarity
function cosineSimilarity(vec1, vec2) {
    let dot = 0;
    let mag1 = 0;
    let mag2 = 0;

    for (let i = 0; i < vec1.length; i++) {
        dot += vec1[i] * vec2[i];
        mag1 += vec1[i] * vec1[i];
        mag2 += vec2[i] * vec2[i];
    }

    return dot / (Math.sqrt(mag1) * Math.sqrt(mag2));
}

// Find nearest neighbors
const queryId = wordToId.get(queryWord);
const queryVec = embeddings[queryId];
const similarities = [];

for (let i = 0; i < embeddings.length; i++) {
    if (i === queryId) continue;

    const sim = cosineSimilarity(queryVec, embeddings[i]);
    similarities.push({ word: idToWord[i], similarity: sim });
}

// Sort by similarity (descending)
similarities.sort((a, b) => b.similarity - a.similarity);

// Display results
console.log(`Most similar words to "${queryWord}":\n`);
for (let i = 0; i < Math.min(topK, similarities.length); i++) {
    const sim = similarities[i].similarity.toFixed(4);
    console.log(`  ${i + 1}. ${similarities[i].word} (${sim})`);
}
console.log('');
