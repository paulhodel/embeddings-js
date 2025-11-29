/**
 * Simple vocabulary analysis - works directly with vocabulary.json
 */

import fs from 'fs';

const vocabFile = './data/vocabulary.json';

console.log('\n' + '='.repeat(70));
console.log('SIMPLE VOCABULARY ANALYSIS');
console.log('='.repeat(70) + '\n');

if (!fs.existsSync(vocabFile)) {
    console.error('Error: vocabulary.json not found!');
    console.log('Run training first: npm run train\n');
    process.exit(1);
}

// Load vocabulary
const vocab = JSON.parse(fs.readFileSync(vocabFile, 'utf-8'));

console.log(`Total words: ${vocab.length}\n`);

// Analyze spectrum sizes
const sizes = vocab
    .filter(w => w.spectrum && w.spectrum.frequencies)
    .map(w => w.spectrum.frequencies.length);

if (sizes.length === 0) {
    console.log('No words have initialized spectra yet!\n');
    process.exit(0);
}

console.log('Spectrum size distribution:');
console.log(`  Min: ${Math.min(...sizes)}`);
console.log(`  Max: ${Math.max(...sizes)}`);
console.log(`  Avg: ${(sizes.reduce((a, b) => a + b, 0) / sizes.length).toFixed(2)}`);
console.log(`  Median: ${sizes.sort((a, b) => a - b)[Math.floor(sizes.length / 2)]}`);

// Count words by spectrum size
const sizeCounts = {};
sizes.forEach(s => sizeCounts[s] = (sizeCounts[s] || 0) + 1);
console.log('\n  Distribution:');
Object.keys(sizeCounts).sort((a, b) => a - b).forEach(size => {
    const count = sizeCounts[size];
    const pct = (count / sizes.length * 100).toFixed(1);
    console.log(`    ${size} freqs: ${count} words (${pct}%)`);
});

// Analyze amplitude distributions
const allAmps = [];
vocab.forEach(w => {
    if (w.spectrum && w.spectrum.amplitudes) {
        allAmps.push(...w.spectrum.amplitudes);
    }
});

if (allAmps.length > 0) {
    allAmps.sort((a, b) => a - b);
    console.log('\nAmplitude distribution:');
    console.log(`  Min: ${allAmps[0].toFixed(6)}`);
    console.log(`  Max: ${allAmps[allAmps.length - 1].toFixed(6)}`);
    console.log(`  Avg: ${(allAmps.reduce((a, b) => a + b, 0) / allAmps.length).toFixed(6)}`);
    console.log(`  Median: ${allAmps[Math.floor(allAmps.length / 2)].toFixed(6)}`);

    // Count zeros
    const zeros = allAmps.filter(a => a === 0).length;
    if (zeros > 0) {
        console.log(`  ⚠️  Zeros: ${zeros} (${(zeros / allAmps.length * 100).toFixed(1)}%)`);
    }
}

// Show some example words
console.log('\n' + '='.repeat(70));
console.log('SAMPLE WORDS');
console.log('='.repeat(70) + '\n');

const wordsWithSpectra = vocab.filter(w => w.spectrum && w.spectrum.frequencies && w.spectrum.frequencies.length > 0);

if (wordsWithSpectra.length > 0) {
    // Show first 10
    console.log('First 10 words with spectra:');
    wordsWithSpectra.slice(0, 10).forEach(w => {
        const freqCount = w.spectrum.frequencies.length;
        const avgAmp = w.spectrum.amplitudes.reduce((a, b) => a + b, 0) / freqCount;
        const maxAmp = Math.max(...w.spectrum.amplitudes);
        console.log(`  "${w.word}": ${freqCount} freqs, avg amp: ${avgAmp.toFixed(4)}, max: ${maxAmp.toFixed(4)}`);
    });

    // Look for specific interesting words
    const interestingWords = ['bank', 'play', 'run', 'light', 'spring', 'the', 'and', 'is'];
    const found = interestingWords
        .map(word => wordsWithSpectra.find(w => w.word === word))
        .filter(w => w);

    if (found.length > 0) {
        console.log('\nInteresting words found:');
        found.forEach(w => {
            const freqCount = w.spectrum.frequencies.length;
            const avgAmp = w.spectrum.amplitudes.reduce((a, b) => a + b, 0) / freqCount;
            const maxAmp = Math.max(...w.spectrum.amplitudes);
            console.log(`  "${w.word}": ${freqCount} freqs, avg amp: ${avgAmp.toFixed(4)}, max: ${maxAmp.toFixed(4)}`);
            console.log(`    Frequencies: [${w.spectrum.frequencies.join(', ')}]`);
            console.log(`    Amplitudes:  [${w.spectrum.amplitudes.map(a => a.toFixed(4)).join(', ')}]`);
        });
    }
}

console.log('\n');
