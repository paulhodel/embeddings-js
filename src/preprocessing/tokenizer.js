/**
 * Text preprocessing and tokenization
 */

export class Tokenizer {
  constructor() {
    this.vocab = new Map(); // word -> id
    this.idToWord = new Map(); // id -> word
    this.wordFreq = new Map(); // word -> frequency
    this.nextId = 0;
  }

  /**
   * Basic tokenization - split on whitespace and punctuation
   */
  tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ') // Replace punctuation with space
      .split(/\s+/)
      .filter(token => token.length > 0);
  }

  /**
   * Build vocabulary from a corpus of texts
   * @param {string[]} texts - array of text documents
   * @param {number} minFreq - minimum frequency to include word
   */
  buildVocab(texts, minFreq = 2) {
    console.log('Building vocabulary...');

    // Count word frequencies
    for (const text of texts) {
      const tokens = this.tokenize(text);
      for (const token of tokens) {
        this.wordFreq.set(token, (this.wordFreq.get(token) || 0) + 1);
      }
    }

    console.log(`Total unique tokens before filtering: ${this.wordFreq.size}`);

    // Filter by minimum frequency and build vocab
    for (const [word, freq] of this.wordFreq.entries()) {
      if (freq >= minFreq) {
        this.vocab.set(word, this.nextId);
        this.idToWord.set(this.nextId, word);
        this.nextId++;
      }
    }

    console.log(`Vocabulary size: ${this.vocab.size}`);
    return this.vocab.size;
  }

  /**
   * Convert word to id
   */
  wordToId(word) {
    return this.vocab.get(word.toLowerCase());
  }

  /**
   * Convert id to word
   */
  idToWord(id) {
    return this.idToWord.get(id);
  }

  /**
   * Check if word is in vocabulary
   */
  hasWord(word) {
    return this.vocab.has(word.toLowerCase());
  }

  /**
   * Get vocabulary size
   */
  get vocabSize() {
    return this.vocab.size;
  }

  /**
   * Convert text to array of word ids
   */
  textToIds(text) {
    const tokens = this.tokenize(text);
    return tokens
      .map(token => this.wordToId(token))
      .filter(id => id !== undefined);
  }
}
