/**
 * ModelPersistence - Save and load trained CSS models
 */

import fs from 'fs';
import path from 'path';

export class ModelPersistence {
  /**
   * Save a trained CSS model to disk
   * @param {CSSTrainer} trainer - Trained CSS trainer
   * @param {Tokenizer} tokenizer - Tokenizer with vocabulary
   * @param {string} filepath - Path to save the model
   */
  static saveModel(trainer, tokenizer, filepath) {
    console.log(`\nSaving model to ${filepath}...`);

    const modelData = {
      version: '1.0.0',
      timestamp: new Date().toISOString(),
      config: trainer.config,
      vocabSize: trainer.vocabSize,

      // Vocabulary
      vocab: Object.fromEntries(tokenizer.vocab),
      wordFreq: Object.fromEntries(tokenizer.wordFreq),

      // Spectra
      spectra: trainer.exportModel().spectra,

      // Statistics
      stats: {
        totalWords: tokenizer.vocab.size,
        avgSparsity: this.calculateAvgSparsity(trainer),
        totalFrequencies: this.countTotalFrequencies(trainer)
      }
    };

    // Create directory if it doesn't exist
    const dir = path.dirname(filepath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    // Save as JSON
    fs.writeFileSync(filepath, JSON.stringify(modelData, null, 2));

    const sizeInMB = (fs.statSync(filepath).size / (1024 * 1024)).toFixed(2);
    console.log(`✓ Model saved (${sizeInMB} MB)`);
    console.log(`  Words: ${modelData.stats.totalWords}`);
    console.log(`  Avg sparsity: ${modelData.stats.avgSparsity.toFixed(2)} frequencies/word`);
  }

  /**
   * Load a trained CSS model from disk
   * @param {string} filepath - Path to the saved model
   * @returns {Object} - { trainer, tokenizer, metadata }
   */
  static loadModel(filepath) {
    console.log(`\nLoading model from ${filepath}...`);

    if (!fs.existsSync(filepath)) {
      throw new Error(`Model file not found: ${filepath}`);
    }

    const modelData = JSON.parse(fs.readFileSync(filepath, 'utf-8'));

    console.log(`✓ Model loaded (version ${modelData.version})`);
    console.log(`  Saved: ${modelData.timestamp}`);
    console.log(`  Words: ${modelData.stats.totalWords}`);

    return {
      modelData,
      config: modelData.config,
      vocab: new Map(Object.entries(modelData.vocab)),
      wordFreq: new Map(Object.entries(modelData.wordFreq)),
      spectra: modelData.spectra,
      stats: modelData.stats
    };
  }

  /**
   * Save training checkpoint
   * @param {Object} checkpoint - Checkpoint data
   * @param {string} filepath - Path to save checkpoint
   */
  static saveCheckpoint(checkpoint, filepath) {
    const dir = path.dirname(filepath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    fs.writeFileSync(filepath, JSON.stringify(checkpoint, null, 2));
    console.log(`✓ Checkpoint saved: ${filepath}`);
  }

  /**
   * Load training checkpoint
   * @param {string} filepath - Path to checkpoint file
   * @returns {Object} - Checkpoint data
   */
  static loadCheckpoint(filepath) {
    if (!fs.existsSync(filepath)) {
      return null;
    }

    const checkpoint = JSON.parse(fs.readFileSync(filepath, 'utf-8'));
    console.log(`✓ Checkpoint loaded from ${checkpoint.timestamp}`);
    return checkpoint;
  }

  /**
   * Calculate average sparsity across all words
   */
  static calculateAvgSparsity(trainer) {
    let totalSparsity = 0;
    let count = 0;

    for (const [wordId, spectrum] of trainer.spectralWord.spectra.entries()) {
      totalSparsity += spectrum.frequencies.length;
      count++;
    }

    return count > 0 ? totalSparsity / count : 0;
  }

  /**
   * Count total number of active frequencies across all words
   */
  static countTotalFrequencies(trainer) {
    let total = 0;

    for (const [wordId, spectrum] of trainer.spectralWord.spectra.entries()) {
      total += spectrum.frequencies.length;
    }

    return total;
  }

  /**
   * Create a backup of an existing model
   */
  static backupModel(filepath) {
    if (!fs.existsSync(filepath)) {
      return;
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupPath = filepath.replace('.json', `.backup.${timestamp}.json`);

    fs.copyFileSync(filepath, backupPath);
    console.log(`✓ Backup created: ${backupPath}`);
  }

  /**
   * List all saved models in a directory
   */
  static listModels(directory = './models') {
    if (!fs.existsSync(directory)) {
      return [];
    }

    const files = fs.readdirSync(directory);
    const models = files
      .filter(f => f.endsWith('.json') && !f.includes('.backup.'))
      .map(f => {
        const filepath = path.join(directory, f);
        const stats = fs.statSync(filepath);
        const data = JSON.parse(fs.readFileSync(filepath, 'utf-8'));

        return {
          filename: f,
          filepath,
          size: (stats.size / (1024 * 1024)).toFixed(2) + ' MB',
          created: data.timestamp,
          vocabSize: data.vocabSize,
          version: data.version
        };
      });

    return models;
  }
}
