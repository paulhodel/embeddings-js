/**
 * HuggingFaceStreamer - Stream data from HuggingFace datasets API
 *
 * Fetches data from karpathy/fineweb-edu-100b-shuffle dataset using the proper API
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
    this.config = 'default';

    // Rate limiting
    this.delayBetweenRequests = 1000; // 1 second between requests
    this.lastRequestTime = 0;

    // Retry configuration
    this.maxRetries = 5;
    this.retryDelay = 2000; // Start with 2 seconds
  }

  /**
   * Sleep for specified milliseconds
   */
  async sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Wait to respect rate limits
   */
  async waitForRateLimit() {
    const now = Date.now();
    const timeSinceLastRequest = now - this.lastRequestTime;

    if (timeSinceLastRequest < this.delayBetweenRequests) {
      const waitTime = this.delayBetweenRequests - timeSinceLastRequest;
      await this.sleep(waitTime);
    }

    this.lastRequestTime = Date.now();
  }

  /**
   * Fetch dataset info to get total number of rows and available configs
   */
  async initialize() {
    try {
      // First, try to get the dataset size
      const sizeUrl = `https://datasets-server.huggingface.co/size?dataset=${this.dataset}`;
      const sizeResponse = await fetch(sizeUrl);

      if (sizeResponse.ok) {
        const data = await sizeResponse.json();

        // Find the train split info
        if (data.size && data.size.splits) {
          const trainSplit = data.size.splits.find(s => s.split === this.split);
          if (trainSplit) {
            this.totalRows = trainSplit.num_rows;
            console.log(`Dataset has ${this.totalRows.toLocaleString()} rows in ${this.split} split`);
          }
        }

        // Get config name if available
        if (data.size && data.size.config) {
          this.config = data.size.config;
        }
      } else {
        console.warn(`Could not fetch dataset size: ${sizeResponse.status}`);
      }

      // Alternatively, try to list parquet files to verify dataset exists
      const parquetUrl = `https://huggingface.co/api/datasets/${this.dataset}/parquet/${this.config}/${this.split}`;
      const parquetResponse = await fetch(parquetUrl);

      if (parquetResponse.ok) {
        const parquetFiles = await parquetResponse.json();
        console.log(`Found ${parquetFiles.length} Parquet files for ${this.dataset}`);
      }

    } catch (error) {
      console.warn(`Could not initialize dataset info: ${error.message}`);
    }
  }

  /**
   * Fetch a batch of rows from the dataset with retry logic
   * @returns {Promise<Array<string>>} Array of text content
   */
  async fetchBatch() {
    if (this.exhausted) {
      return [];
    }

    let lastError = null;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        // Wait to respect rate limits
        await this.waitForRateLimit();

        // Properly encode the dataset name for the URL
        const encodedDataset = encodeURIComponent(this.dataset);
        const url = `${this.baseUrl}?dataset=${encodedDataset}&config=${this.config}&split=${this.split}&offset=${this.offset}&length=${this.batchSize}`;

        console.log(`Fetching: offset=${this.offset}, length=${this.batchSize}`);

        const response = await fetch(url);

        // Handle rate limiting (429)
        if (response.status === 429) {
          const retryAfter = response.headers.get('retry-after');
          const waitTime = retryAfter ? parseInt(retryAfter) * 1000 : this.retryDelay * Math.pow(2, attempt);

          console.warn(`⚠️  Rate limited (429). Waiting ${Math.round(waitTime / 1000)}s before retry ${attempt + 1}/${this.maxRetries}...`);
          await this.sleep(waitTime);
          continue; // Retry
        }

        if (!response.ok) {
          if (response.status === 404) {
            // Try to get more details about the error
            const errorText = await response.text();
            console.error(`404 Error details: ${errorText}`);
            console.log('Reached end of dataset or dataset not found');
            this.exhausted = true;
            return [];
          }

          // Get error details for other status codes
          const errorText = await response.text();
          throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
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
        lastError = error;

        // If it's a network error, retry with exponential backoff
        if (attempt < this.maxRetries - 1) {
          const waitTime = this.retryDelay * Math.pow(2, attempt);
          console.warn(`⚠️  Error: ${error.message}. Retrying in ${Math.round(waitTime / 1000)}s (${attempt + 1}/${this.maxRetries})...`);
          await this.sleep(waitTime);
        }
      }
    }

    // All retries exhausted
    console.error(`❌ Failed after ${this.maxRetries} attempts: ${lastError.message}`);
    throw lastError;
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
