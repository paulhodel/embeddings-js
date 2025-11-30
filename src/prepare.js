/**
 * Vocabulary Preparation - Streaming Version
 * Process one parquet file at a time - memory efficient
 * Count frequencies in single pass - no separate loops
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

const CONFIG = {
    parquetDir: './data/parquet',
    outputFile: './data/dictionary.ndjson',
    statsFile: './data/vocab_stats.json',
    vocabularySize: 50000,
    embeddingDim: 64,
    decimalPrecision: 4,
};

async function main() {
    console.log('\n' + '='.repeat(60));
    console.log('VOCABULARY PREPARATION - STREAMING');
    console.log('='.repeat(60));
    console.log('\nConfiguration:');
    console.log(`  Vocabulary size: ${CONFIG.vocabularySize}`);
    console.log(`  Embedding dim:   ${CONFIG.embeddingDim}`);
    console.log(`  Decimal places:  ${CONFIG.decimalPrecision}`);
    console.log(`  Parquet dir:     ${CONFIG.parquetDir}`);
    console.log(`  Output file:     ${CONFIG.outputFile}`);

    const startTime = Date.now();

    // ============================================
    // STREAM PARQUET FILES - ONE AT A TIME
    // Count word frequencies in single pass
    // ============================================
    console.log(`\nStreaming parquet files from: ${CONFIG.parquetDir}`);

    const parquetFiles = fs.readdirSync(CONFIG.parquetDir)
        .filter(f => f.endsWith('.parquet'))
        .sort();

    console.log(`Found ${parquetFiles.length} parquet files`);

    const wordCounts = new Map();
    let totalTokens = 0;
    let totalDocuments = 0;

    const numFiles = parquetFiles.length;
    for (let fileIdx = 0; fileIdx < numFiles; fileIdx++) {
        const filepath = path.join(CONFIG.parquetDir, parquetFiles[fileIdx]);
        console.log(`\n[${fileIdx + 1}/${numFiles}] Processing ${parquetFiles[fileIdx]}...`);

        // Load one parquet file
        let documents;
        try {
            const pythonCmd = `python scripts/read_parquet.py "${filepath}"`;
            const output = execSync(pythonCmd, {
                encoding: 'utf8',
                maxBuffer: 500 * 1024 * 1024
            });
            documents = JSON.parse(output);
            console.log(`  Loaded ${documents.length} documents`);
        } catch (error) {
            console.error(`  Error reading ${filepath}:`, error.message);
            continue;
        }

        // Tokenize and count in single pass
        console.log('  Tokenizing and counting...');
        let currentWord = '';
        let hasLetter = false;

        const numDocs = documents.length;
        for (let docIdx = 0; docIdx < numDocs; docIdx++) {
            const text = documents[docIdx];
            const textLen = text.length;

            // Character-by-character processing
            for (let charIdx = 0; charIdx < textLen; charIdx++) {
                const code = text.charCodeAt(charIdx);

                // Letters: A-Z (65-90), a-z (97-122)
                if ((code >= 65 && code <= 90) || (code >= 97 && code <= 122)) {
                    if (code >= 65 && code <= 90) {
                        currentWord += String.fromCharCode(code + 32);
                    } else {
                        currentWord += text[charIdx];
                    }
                    hasLetter = true;

                // Numbers: 0-9 (48-57)
                } else if (code >= 48 && code <= 57) {
                    currentWord += text[charIdx];

                // Any other character - end current word
                } else {
                    if (hasLetter && currentWord.length >= 2) {
                        const count = wordCounts.get(currentWord);
                        wordCounts.set(currentWord, (count || 0) + 1);
                        totalTokens++;
                    }
                    currentWord = '';
                    hasLetter = false;
                }
            }

            // End of document - finalize last word
            if (hasLetter && currentWord.length >= 2) {
                const count = wordCounts.get(currentWord);
                wordCounts.set(currentWord, (count || 0) + 1);
                totalTokens++;
            }
            currentWord = '';
            hasLetter = false;
        }

        totalDocuments += numDocs;
        console.log(`  Processed: ${numDocs} docs, ${totalTokens} total tokens so far`);
        console.log(`  Unique words so far: ${wordCounts.size}`);

        // Free memory
        documents = null;
    }

    console.log(`\n${'='.repeat(60)}`);
    console.log('Streaming complete!');
    console.log(`Total documents: ${totalDocuments}`);
    console.log(`Total tokens: ${totalTokens}`);
    console.log(`Unique words: ${wordCounts.size}`);
    console.log('='.repeat(60));

    if (wordCounts.size === 0) {
        console.error('\nError: No words found in parquet files');
        process.exit(1);
    }

    // ============================================
    // SELECT TOP K WORDS
    // ============================================
    console.log(`\nSelecting top ${CONFIG.vocabularySize} most frequent words...`);

    const topWords = [];
    for (const [word, frequency] of wordCounts) {
        topWords.push({ word, frequency });
    }
    topWords.sort((a, b) => b.frequency - a.frequency);
    topWords.length = Math.min(CONFIG.vocabularySize, topWords.length);

    console.log(`Selected ${topWords.length} words`);
    console.log(`Most frequent: "${topWords[0].word}" (${topWords[0].frequency} occurrences)`);
    console.log(`Least frequent in vocab: "${topWords[topWords.length - 1].word}" (${topWords[topWords.length - 1].frequency} occurrences)`);

    // Free memory
    wordCounts.clear();

    // ============================================
    // INITIALIZE EMBEDDINGS
    // ============================================
    console.log(`\nInitializing embeddings (${CONFIG.embeddingDim} dimensions)...`);

    const embDim = CONFIG.embeddingDim;
    const numWords = topWords.length;

    for (let i = 0; i < numWords; i++) {
        // Generate uniform random vector [-0.1, 0.1]
        const vector = new Array(embDim);
        for (let j = 0; j < embDim; j++) {
            vector[j] = (Math.random() * 0.2) - 0.1;
        }

        topWords[i].id = i;
        topWords[i].vector = vector;
    }

    console.log(`Embeddings initialized for ${topWords.length} words`);

    // ============================================
    // CALCULATE STATISTICS
    // ============================================
    console.log('\nCalculating coverage statistics...');

    let tokensCovered = 0;
    const vocabSize = topWords.length;
    for (let i = 0; i < vocabSize; i++) {
        tokensCovered += topWords[i].frequency;
    }

    const oovTokens = totalTokens - tokensCovered;
    const coveragePercent = (tokensCovered / totalTokens) * 100;
    const oovPercent = (oovTokens / totalTokens) * 100;
    const estimatedWindowLoss = oovPercent * 2.5;

    const stats = {
        vocabularySize: topWords.length,
        totalTokens: totalTokens,
        totalDocuments: totalDocuments,
        uniqueWords: topWords.length,
        tokensCovered: tokensCovered,
        coveragePercent: parseFloat(coveragePercent.toFixed(2)),
        oovTokens: oovTokens,
        oovPercent: parseFloat(oovPercent.toFixed(2)),
        estimatedWindowLoss: parseFloat(estimatedWindowLoss.toFixed(2))
    };

    console.log(`Coverage: ${coveragePercent.toFixed(2)}%`);
    console.log(`OOV rate: ${oovPercent.toFixed(2)}%`);

    // ============================================
    // SAVE DICTIONARY TO NDJSON
    // ============================================
    console.log(`\nSaving dictionary to: ${CONFIG.outputFile}`);

    const dir = path.dirname(CONFIG.outputFile);
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }

    const writeStream = fs.createWriteStream(CONFIG.outputFile);

    const numWordsToSave = topWords.length;
    const precision = CONFIG.decimalPrecision;

    for (let i = 0; i < numWordsToSave; i++) {
        const item = topWords[i];
        const vec = item.vector;
        const vecLen = vec.length;

        // Build vector string
        let vectorStr = '';
        for (let j = 0; j < vecLen; j++) {
            if (j > 0) vectorStr += ' ';
            vectorStr += vec[j].toFixed(precision);
        }

        const line = `${item.word}\t${item.id}\t${item.frequency}\t${vectorStr}\n`;
        writeStream.write(line);

        if ((i + 1) % 10000 === 0) {
            console.log(`  Saved ${i + 1}/${numWordsToSave} words`);
        }
    }

    writeStream.end();

    // Wait for write to complete
    await new Promise((resolve, reject) => {
        writeStream.on('finish', resolve);
        writeStream.on('error', reject);
    });

    const fileStats = fs.statSync(CONFIG.outputFile);
    const sizeMB = (fileStats.size / 1024 / 1024).toFixed(1);
    console.log(`Dictionary saved (${sizeMB}MB)`);

    // ============================================
    // SAVE STATISTICS
    // ============================================
    console.log(`\nSaving statistics to: ${CONFIG.statsFile}`);

    const statsDir = path.dirname(CONFIG.statsFile);
    if (!fs.existsSync(statsDir)) {
        fs.mkdirSync(statsDir, { recursive: true });
    }

    fs.writeFileSync(CONFIG.statsFile, JSON.stringify(stats, null, 2));
    console.log('Statistics saved');

    // ============================================
    // COMPLETE
    // ============================================
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    console.log('\n' + '='.repeat(60));
    console.log('PREPARATION COMPLETE!');
    console.log('='.repeat(60));
    console.log(`Time elapsed: ${elapsed}s`);
    console.log(`Total documents: ${totalDocuments}`);
    console.log(`Total tokens: ${totalTokens}`);
    console.log(`Vocabulary: ${topWords.length} words`);
    console.log(`Coverage: ${coveragePercent.toFixed(2)}%`);
    console.log(`\nNext step: Run training with "node src/train2.js"`);
    console.log('');
}

main().catch(error => {
    console.error('\nPreparation failed:', error.message);
    console.error(error.stack);
    process.exit(1);
});
