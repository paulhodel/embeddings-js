import fs from 'fs';
import path from 'path';

const CONFIG = {
    vocabularyFile: './data/vocabulary.json',
}

// vocabulary: Map<word, {id, spectrum}>
const vocabulary = new Map();
// cache: Map<id, {id, spectrum}> - points to same objects as vocabulary
const cache = new Map();

/**
 * Save the vocabulary to a JSON file
 */
function saveVocabulary() {
    if (vocabulary.size === 0) {
        console.log('Vocabulary is empty, nothing to save.');
        return;
    }

    // Create directory if it doesn't exist
    const dir = path.dirname(CONFIG.vocabularyFile);
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }

    // Convert Map to array of word objects for JSON serialization
    const vocabArray = Array.from(vocabulary.entries()).map(([word, obj]) => ({
        word,
        id: obj.id,
        spectrum: obj.spectrum,
        frequency: obj.frequency || 0  // Include frequency if available
    }));

    // Save to file
    fs.writeFileSync(CONFIG.vocabularyFile, JSON.stringify(vocabArray, null, 2));
    console.log(`✓ Vocabulary saved to ${CONFIG.vocabularyFile}`);
    console.log(`  Total words: ${vocabulary.size}`);
}

/**
 * Load vocabulary from JSON file
 * @returns {boolean} - True if loaded successfully, false otherwise
 */
function loadVocabulary() {
    if (!fs.existsSync(CONFIG.vocabularyFile)) {
        console.log('No vocabulary file found, starting with empty vocabulary.');
        return false;
    }

    try {
        // Read and parse JSON file
        const data = fs.readFileSync(CONFIG.vocabularyFile, 'utf-8');
        const vocabArray = JSON.parse(data);

        // Clear existing maps and load from file
        vocabulary.clear();
        cache.clear();

        for (const item of vocabArray) {
            // Create the word object
            const wordObj = {
                id: item.id,
                spectrum: item.spectrum || [],
                frequency: item.frequency || 0
            };

            // Add to both maps (pointing to the same object)
            vocabulary.set(item.word, wordObj);
            cache.set(item.id, wordObj);
        }

        console.log(`✓ Vocabulary loaded from ${CONFIG.vocabularyFile}`);
        console.log(`  Total words: ${vocabulary.size}`);
        return true;
    } catch (error) {
        console.error(`Error loading vocabulary: ${error.message}`);
        return false;
    }
}

/**
 * Get word object by its ID
 * @param {number} wordId - The word ID
 * @returns {Object|null} - The word object {id, spectrum}, or null if not found
 */
function getWordById(wordId) {
    return cache.get(wordId) || null;
}

/**
 * Get word object by word string
 * @param {string} word - The word string
 * @returns {Object|undefined} - The word object {id, spectrum}, or undefined if not found
 */
function getWordByString(word) {
    return vocabulary.get(word);
}

/**
 * Get ID by word string
 * @param {string} word - The word string
 * @returns {number|undefined} - The word ID, or undefined if not found
 */
function getIdByWord(word) {
    const wordObj = vocabulary.get(word);
    return wordObj ? wordObj.id : undefined;
}

/**
 * Add a word to the vocabulary
 * @param {string} word - The word to add
 * @returns {Object} - The word object {id, spectrum}
 */
function addWord(word) {
    if (vocabulary.has(word)) {
        return vocabulary.get(word);
    }

    // Create new word object
    const wordObj = {
        id: vocabulary.size,
        spectrum: [],
        frequency: 0
    };

    // Add to both maps (same object reference)
    vocabulary.set(word, wordObj);
    cache.set(wordObj.id, wordObj);

    return wordObj;
}

/**
 * Check if a word exists in vocabulary
 * @param {string} word - The word to check
 * @returns {boolean} - True if word exists
 */
function hasWord(word) {
    return vocabulary.has(word);
}

/**
 * Get the vocabulary size
 * @returns {number} - Number of words in vocabulary
 */
function getVocabSize() {
    return vocabulary.size;
}

/**
 * Clear the vocabulary
 */
function clearVocabulary() {
    vocabulary.clear();
    cache.clear();
}

/**
 * Initialize the vocabulary module
 * Loads existing vocabulary from file if available
 */
function init() {
    console.log('Initializing vocabulary module...');
    loadVocabulary();
}

/**
 * Get word string by ID (reverse lookup)
 * @param {number} wordId - The word ID
 * @returns {string|null} - The word string, or null if not found
 */
function getWordString(wordId) {
    for (const [word, wordObj] of vocabulary.entries()) {
        if (wordObj.id === wordId) {
            return word;
        }
    }
    return null;
}

/**
 * Get all words as array of {word, id, spectrum}
 * @returns {Array} - Array of word objects
 */
function getAllWords() {
    const words = [];
    for (const [word, wordObj] of vocabulary.entries()) {
        words.push({
            word,
            id: wordObj.id,
            spectrum: wordObj.spectrum,
            frequency: wordObj.frequency || 0
        });
    }
    return words;
}

export {
    init,
    getWordById,
    getWordByString,
    getWordString,
    getIdByWord,
    addWord,
    hasWord,
    getVocabSize,
    saveVocabulary,
    loadVocabulary,
    clearVocabulary,
    getAllWords
}