/**
 * HuggingFaceStreamer - Stream data from HuggingFace datasets API
 *
 * Fetches data from karpathy/fineweb-edu-100b-shuffle dataset
 */

import fetch from 'node-fetch';

export class HuggingFaceStreamer {
  /**
   * @param {string} dataset - Dataset name
   * @param {string} split - Dataset split (train, validation, test)
   * @param {number} batchSize - Number of examples per batch
   */
  constructor(dataset = 'karpathy/fineweb-edu-100b-shuffle', split = 'train', batchSize = 100) {
    this.dataset = dataset;
    this.split = split;
    this.batchSize = batchSize;
    this.baseUrl = 'https://datasets-server.huggingface.co/rows';
    this.offset = 0;
    this.totalRows = null;
    this.exhausted = false;
  }

  /**
   * Fetch dataset info to get total number of rows
   */
  async initialize() {
    try {
      const url = `https://datasets-server.huggingface.co/size?dataset=${this.dataset}`;
      const response = await fetch(url);

      if (!response.ok) {
        console.warn(`Could not fetch dataset size: ${response.status}`);
        return;
      }

      const data = await response.json();

      // Find the train split info
      if (data.size && data.size.splits) {
        const trainSplit = data.size.splits.find(s => s.split === this.split);
        if (trainSplit) {
          this.totalRows = trainSplit.num_rows;
          console.log(`Dataset has ${this.totalRows.toLocaleString()} rows in ${this.split} split`);
        }
      }
    } catch (error) {
      console.warn(`Could not initialize dataset info: ${error.message}`);
    }
  }

  /**
   * Fetch a batch of rows from the dataset
   * @returns {Promise<Array<string>>} Array of text content
   */
  async fetchBatch() {
    if (this.exhausted) {
      return [];
    }

    try {
      const url = `${this.baseUrl}?dataset=${this.dataset}&config=default&split=${this.split}&offset=${this.offset}&length=${this.batchSize}`;

      const response = await fetch(url);

      if (!response.ok) {
        if (response.status === 404) {
          console.warn('Reached end of dataset or dataset not found');
          this.exhausted = true;
          return [];
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (!data.rows || data.rows.length === 0) {
        console.log('No more rows available');
        this.exhausted = true;
        return [];
      }

      // Extract text from rows
      const texts = data.rows.map(row => {
        // The fineweb dataset has a 'text' field
        if (row.row && row.row.text) {
          return row.row.text;
        }
        return null;
      }).filter(text => text !== null && text.length > 0);

      this.offset += data.rows.length;

      return texts;
    } catch (error) {
      console.error(`Error fetching batch: ${error.message}`);
      throw error;
    }
  }

  /**
   * Stream batches of text data
   * @param {number} maxBatches - Maximum number of batches to fetch (null = unlimited)
   * @returns {AsyncGenerator<Array<string>>}
   */
  async *stream(maxBatches = null) {
    let batchCount = 0;

    while (!this.exhausted) {
      if (maxBatches !== null && batchCount >= maxBatches) {
        break;
      }

      const batch = await this.fetchBatch();

      if (batch.length === 0) {
        break;
      }

      yield batch;
      batchCount++;
    }
  }

  /**
   * Get progress information
   */
  getProgress() {
    return {
      processed: this.offset,
      total: this.totalRows,
      percentage: this.totalRows ? (this.offset / this.totalRows * 100).toFixed(2) : null,
      exhausted: this.exhausted
    };
  }

  /**
   * Reset the streamer to start from beginning
   */
  reset() {
    this.offset = 0;
    this.exhausted = false;
  }
}
