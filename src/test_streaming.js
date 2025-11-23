/**
 * Test HuggingFace streaming functionality
 * This is a quick test to verify data can be fetched from HuggingFace
 */

import { HuggingFaceStreamer } from './data/HuggingFaceStreamer.js';

async function testStreaming() {
  console.log('\n========================================');
  console.log('TESTING HUGGINGFACE STREAMING');
  console.log('========================================\n');

  // Create streamer
  const streamer = new HuggingFaceStreamer(
    'karpathy/fineweb-edu-100b-shuffle',
    'train',
    10  // Small batch size for testing
  );

  // Initialize (get dataset info)
  console.log('Initializing dataset...');
  await streamer.initialize();

  // Fetch a few batches
  console.log('\nFetching 3 test batches...\n');

  let batchCount = 0;
  let totalDocs = 0;

  for await (const batch of streamer.stream(3)) {
    batchCount++;
    totalDocs += batch.length;

    console.log(`\nBatch ${batchCount}:`);
    console.log(`  Documents: ${batch.length}`);

    if (batch.length > 0) {
      console.log(`  First doc preview: ${batch[0].substring(0, 100)}...`);
      console.log(`  First doc length: ${batch[0].length} chars`);
    }

    const progress = streamer.getProgress();
    console.log(`  Progress: ${progress.processed} rows processed`);
  }

  console.log('\n========================================');
  console.log('STREAMING TEST COMPLETE');
  console.log('========================================');
  console.log(`Total batches: ${batchCount}`);
  console.log(`Total documents: ${totalDocs}`);
  console.log('✓ Streaming works!\n');
}

testStreaming().catch(error => {
  console.error('\n❌ Test failed:');
  console.error(error);
  process.exit(1);
});
