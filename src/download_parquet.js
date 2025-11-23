/**
 * Simple Parquet Downloader - Direct HTTPS download from HuggingFace
 *
 * Usage: node src/download_parquet.js
 *
 * Downloads raw Parquet files directly from HuggingFace dataset.
 * No API, no rate limits, just direct file downloads.
 */

import fs from 'fs';
import https from 'https';
import path from 'path';

// ============================================
// CONFIGURATION
// ============================================

const CONFIG = {
    // Dataset info
    dataset: 'karpathy/fineweb-edu-100b-shuffle',

    // Which files to download (each file is ~93MB)
    numShards: 5,       // Download 5 files (~465MB total)

    // Output directory
    outputDir: './data/parquet',

    // Progress tracking file
    progressFile: './data/download_progress.json',
};

// ============================================
// DOWNLOAD FUNCTION
// ============================================

/**
 * Download a single Parquet shard file
 */
async function downloadShard(shardIndex) {
    const shardName = `shard_${shardIndex.toString().padStart(5, '0')}.parquet`;
    const url = `https://huggingface.co/datasets/${CONFIG.dataset}/resolve/main/${shardName}?download=true`;
    const filepath = path.join(CONFIG.outputDir, shardName);

    // Check if already downloaded
    if (fs.existsSync(filepath)) {
        const stats = fs.statSync(filepath);
        console.log(`✓ Already downloaded: ${shardName} (${(stats.size / 1024 / 1024).toFixed(1)}MB)`);
        return filepath;
    }

    // Ensure directory exists
    if (!fs.existsSync(CONFIG.outputDir)) {
        fs.mkdirSync(CONFIG.outputDir, {recursive: true});
    }

    console.log(`\nDownloading: ${shardName}`);
    console.log(`URL: ${url}`);

    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(filepath);
        let downloadedBytes = 0;
        let lastLogTime = Date.now();

        https.get(url, {
            headers: {
                'User-Agent': 'Mozilla/5.0'
            }
        }, (response) => {
            // Handle redirects
            if (response.statusCode === 302 || response.statusCode === 301) {
                const redirectUrl = response.headers.location;
                console.log(`Following redirect...`);

                https.get(redirectUrl, (redirectResponse) => {
                    if (redirectResponse.statusCode !== 200) {
                        file.close();
                        fs.unlinkSync(filepath);
                        reject(new Error(`HTTP ${redirectResponse.statusCode}: ${redirectResponse.statusMessage}`));
                        return;
                    }

                    const totalBytes = parseInt(redirectResponse.headers['content-length'], 10);
                    console.log(`File size: ${(totalBytes / 1024 / 1024).toFixed(1)}MB`);

                    redirectResponse.on('data', (chunk) => {
                        downloadedBytes += chunk.length;

                        // Log progress every 2 seconds
                        const now = Date.now();
                        if (now - lastLogTime > 2000) {
                            const percent = ((downloadedBytes / totalBytes) * 100).toFixed(1);
                            const mbDownloaded = (downloadedBytes / 1024 / 1024).toFixed(1);
                            const mbTotal = (totalBytes / 1024 / 1024).toFixed(1);
                            const speed = ((downloadedBytes / 1024 / 1024) / ((now - Date.now() + 2000) / 1000)).toFixed(1);
                            process.stdout.write(`\r  Progress: ${percent}% (${mbDownloaded}MB / ${mbTotal}MB) @ ${speed}MB/s`);
                            lastLogTime = now;
                        }
                    });

                    redirectResponse.pipe(file);

                    file.on('finish', () => {
                        file.close();
                        console.log(`\n✓ Downloaded: ${shardName} (${(downloadedBytes / 1024 / 1024).toFixed(1)}MB)`);
                        resolve(filepath);
                    });
                }).on('error', (err) => {
                    file.close();
                    fs.unlinkSync(filepath);
                    reject(err);
                });

                return;
            }

            if (response.statusCode !== 200) {
                file.close();
                if (fs.existsSync(filepath)) {
                    fs.unlinkSync(filepath);
                }
                const errorMsg = `HTTP ${response.statusCode}: ${response.statusMessage || 'Unknown error'}`;
                console.error(`\n${errorMsg}`);
                reject(new Error(errorMsg));
                return;
            }

            const totalBytes = parseInt(response.headers['content-length'], 10);
            console.log(`File size: ${(totalBytes / 1024 / 1024).toFixed(1)}MB`);

            response.on('data', (chunk) => {
                downloadedBytes += chunk.length;

                // Log progress every 2 seconds
                const now = Date.now();
                if (now - lastLogTime > 2000) {
                    const percent = ((downloadedBytes / totalBytes) * 100).toFixed(1);
                    const mbDownloaded = (downloadedBytes / 1024 / 1024).toFixed(1);
                    const mbTotal = (totalBytes / 1024 / 1024).toFixed(1);
                    process.stdout.write(`\r  Progress: ${percent}% (${mbDownloaded}MB / ${mbTotal}MB)`);
                    lastLogTime = now;
                }
            });

            response.pipe(file);

            file.on('finish', () => {
                file.close();
                console.log(`\n✓ Downloaded: ${shardName} (${(downloadedBytes / 1024 / 1024).toFixed(1)}MB)`);
                resolve(filepath);
            });
        }).on('error', (err) => {
            file.close();
            if (fs.existsSync(filepath)) {
                fs.unlinkSync(filepath);
            }
            reject(err);
        });
    });
}

// ============================================
// PROGRESS TRACKING
// ============================================

/**
 * Load download progress from file
 */
function loadProgress() {
    try {
        if (fs.existsSync(CONFIG.progressFile)) {
            const data = JSON.parse(fs.readFileSync(CONFIG.progressFile, 'utf8'));
            return data;
        }
    } catch (error) {
        console.warn(`Warning: Could not load progress file: ${error.message}`);
    }

    return {
        lastDownloadedShard: -1,
        totalDownloaded: 0,
        downloadedFiles: []
    };
}

/**
 * Save download progress to file
 */
function saveProgress(progress) {
    try {
        const dir = path.dirname(CONFIG.progressFile);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, {recursive: true});
        }
        fs.writeFileSync(CONFIG.progressFile, JSON.stringify(progress, null, 2));
    } catch (error) {
        console.warn(`Warning: Could not save progress: ${error.message}`);
    }
}

/**
 * Get the next shard index to download
 */
function getNextShardIndex(progress) {
    return progress.lastDownloadedShard + 1;
}

// ============================================
// MAIN FUNCTION
// ============================================

async function main() {
    console.log('\n' + '='.repeat(70));
    console.log('PARQUET FILE DOWNLOADER');
    console.log('='.repeat(70) + '\n');

    // Load progress
    const progress = loadProgress();
    const startShard = getNextShardIndex(progress);

    console.log('Configuration:');
    console.log(`  Dataset: ${CONFIG.dataset}`);
    console.log(`  Shards: ${startShard} to ${startShard + CONFIG.numShards - 1} (${CONFIG.numShards} files)`);
    console.log(`  Estimated size: ~${(CONFIG.numShards * 93).toFixed(0)}MB`);
    console.log(`  Output directory: ${CONFIG.outputDir}`);

    if (progress.lastDownloadedShard >= 0) {
        console.log(`\n  Previous downloads: ${progress.downloadedFiles.length} files`);
        console.log(`  Last shard: ${progress.lastDownloadedShard}`);
        console.log(`  Resuming from shard: ${startShard}`);
    }
    console.log('');

    const startTime = Date.now();
    let totalBytes = 0;
    let newDownloads = 0;

    for (let i = 0; i < CONFIG.numShards; i++) {
        const shardIndex = startShard + i;
        console.log(`\n[${i + 1}/${CONFIG.numShards}] Downloading shard ${shardIndex}...`);

        try {
            const filepath = await downloadShard(shardIndex);

            // Get file size
            if (fs.existsSync(filepath)) {
                const stats = fs.statSync(filepath);
                totalBytes += stats.size;

                // Update progress only if newly downloaded
                const shardName = `shard_${shardIndex.toString().padStart(5, '0')}.parquet`;
                if (!progress.downloadedFiles.includes(shardName)) {
                    progress.downloadedFiles.push(shardName);
                    newDownloads++;
                }

                progress.lastDownloadedShard = shardIndex;
                progress.totalDownloaded = progress.downloadedFiles.length;

                // Save progress after each successful download
                saveProgress(progress);
            }

        } catch (error) {
            console.error(`\n❌ Error downloading shard ${shardIndex}:`);
            console.error(`   Message: ${error.message}`);
            if (error.code) {
                console.error(`   Code: ${error.code}`);
            }
            if (error.stack) {
                console.error(`   Stack: ${error.stack}`);
            }
            console.error('\nProgress has been saved. Run again to resume.');
            process.exit(1);
        }
    }

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    const avgSpeed = totalBytes > 0 ? (totalBytes / 1024 / 1024 / elapsed).toFixed(1) : '0';

    console.log('\n' + '='.repeat(70));
    console.log('DOWNLOAD COMPLETE!');
    console.log('='.repeat(70));
    console.log(`\nNew downloads: ${newDownloads} files (${(totalBytes / 1024 / 1024).toFixed(1)}MB)`);
    console.log(`Total downloads: ${progress.downloadedFiles.length} files`);
    console.log(`Time elapsed: ${elapsed}s`);
    console.log(`Average speed: ${avgSpeed}MB/s`);
    console.log(`Files saved to: ${CONFIG.outputDir}`);
    console.log(`\nNext shard to download: ${progress.lastDownloadedShard + 1}`);
    console.log('\nRun again to download the next batch!');
    console.log('\nNext steps:');
    console.log('  1. Install parquet reading library: npm install parquetjs');
    console.log('  2. Process the files to extract text data');
    console.log('  3. Run training\n');
}

// ============================================
// RUN
// ============================================

main().catch(error => {
    console.error('\n❌ Download failed:', error.message);
    process.exit(1);
});
