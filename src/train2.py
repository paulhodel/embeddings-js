"""
Training Script - Model 2 (Skip-gram with Negative Sampling)
Optimized PyTorch/CUDA version with proper GPU utilization
- Large batch processing for GPU efficiency
- Vectorized operations with PyTorch tensors
- One-file-at-a-time memory management
"""

import json
import os
import time
import torch
import pandas as pd

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    # Data
    'parquet_dir': './data/parquet',
    'dictionary_file': './data/dictionary.ndjson',
    'checkpoint_dir': './data/checkpoints2',
    'max_parquet_files': 10,

    # Architecture
    'embedding_dim': 64,
    'context_window': 3,

    # Training
    'learning_rate': 0.025,
    'batch_size': 2048,  # Large batches for GPU
    'epochs': 5,
    'negative_samples': 5,

    # Checkpointing
    'checkpoint_every': 100000,  # In pairs
}


def tokenize_text(text, word_to_id):
    """Fast tokenization matching JavaScript version"""
    tokens = []
    current_word = []
    has_letter = False

    for char in text:
        code = ord(char)

        if (65 <= code <= 90) or (97 <= code <= 122):
            if 65 <= code <= 90:
                current_word.append(chr(code + 32))
            else:
                current_word.append(char)
            has_letter = True
        elif 48 <= code <= 57:
            current_word.append(char)
        else:
            if has_letter and len(current_word) >= 2:
                word = ''.join(current_word)
                word_id = word_to_id.get(word)
                if word_id is not None:
                    tokens.append(word_id)
            current_word = []
            has_letter = False

    if has_letter and len(current_word) >= 2:
        word = ''.join(current_word)
        word_id = word_to_id.get(word)
        if word_id is not None:
            tokens.append(word_id)

    return tokens


def extract_training_pairs(tokenized_docs, context_window):
    """Extract all (center, context) pairs from tokenized documents"""
    pairs = []

    for tokens in tokenized_docs:
        num_tokens = len(tokens)
        for center_idx in range(context_window, num_tokens - context_window):
            center_id = tokens[center_idx]

            for offset in range(-context_window, context_window + 1):
                if offset == 0:
                    continue
                context_id = tokens[center_idx + offset]
                pairs.append((center_id, context_id))

    return pairs


def train_batch_gpu(center_ids, context_ids, input_embeddings, output_embeddings,
                    neg_sampling_probs, vocab_size, lr, neg_samples, device):
    """
    Vectorized batch training with negative sampling on GPU

    Args:
        center_ids: Tensor of shape (batch_size,)
        context_ids: Tensor of shape (batch_size,)
        input_embeddings: Tensor of shape (vocab_size, emb_dim)
        output_embeddings: Tensor of shape (vocab_size, emb_dim)
        neg_sampling_probs: Tensor of shape (vocab_size,)
        vocab_size: Size of vocabulary
        lr: Learning rate
        neg_samples: Number of negative samples
        device: torch device

    Returns:
        loss: Scalar loss for this batch
    """
    batch_size = center_ids.shape[0]
    emb_dim = input_embeddings.shape[1]

    # Get embeddings for batch
    center_vecs = input_embeddings[center_ids]  # (batch_size, emb_dim)
    context_vecs = output_embeddings[context_ids]  # (batch_size, emb_dim)

    # ========================================
    # POSITIVE SAMPLES
    # ========================================
    pos_dots = (center_vecs * context_vecs).sum(dim=1)  # (batch_size,)
    pos_probs = torch.sigmoid(pos_dots)
    pos_loss = -torch.log(pos_probs + 1e-8)

    # Gradients for positive samples
    pos_grad = (pos_probs - 1.0).unsqueeze(1)  # (batch_size, 1)

    # ========================================
    # NEGATIVE SAMPLES
    # ========================================
    # Sample negatives for entire batch at once
    neg_ids = torch.multinomial(
        neg_sampling_probs,
        batch_size * neg_samples,
        replacement=True
    ).reshape(batch_size, neg_samples)

    # Get negative embeddings: (batch_size, neg_samples, emb_dim)
    neg_vecs = output_embeddings[neg_ids]

    # Compute all negative dot products
    center_vecs_expanded = center_vecs.unsqueeze(1)  # (batch_size, 1, emb_dim)
    neg_dots = torch.bmm(
        center_vecs_expanded,
        neg_vecs.transpose(1, 2)
    ).squeeze(1)  # (batch_size, neg_samples)

    neg_probs = torch.sigmoid(neg_dots)
    neg_loss = -torch.log(1.0 - neg_probs + 1e-8).sum(dim=1)  # (batch_size,)

    # Gradients for negative samples
    neg_grad = neg_probs.unsqueeze(2)  # (batch_size, neg_samples, 1)

    # ========================================
    # UPDATE EMBEDDINGS
    # ========================================
    with torch.no_grad():
        # Input gradients (center words)
        input_grad = pos_grad * context_vecs  # (batch_size, emb_dim)
        neg_contribution = (neg_grad * neg_vecs).sum(dim=1)  # (batch_size, emb_dim)
        input_grad += neg_contribution

        # Update input embeddings
        input_embeddings.index_add_(0, center_ids, -lr * input_grad)

        # Normalize updated center embeddings
        updated_norms = torch.norm(input_embeddings[center_ids], dim=1, keepdim=True) + 1e-8
        input_embeddings[center_ids] /= updated_norms

        # Update output embeddings (context)
        context_grad = pos_grad * center_vecs
        output_embeddings.index_add_(0, context_ids, -lr * context_grad)

        # Update output embeddings (negatives)
        neg_ids_flat = neg_ids.reshape(-1)
        center_vecs_repeated = center_vecs.unsqueeze(1).expand(-1, neg_samples, -1)
        neg_grads_flat = (neg_grad.squeeze(2).unsqueeze(2) * center_vecs_repeated).reshape(-1, emb_dim)
        output_embeddings.index_add_(0, neg_ids_flat, -lr * neg_grads_flat)

    # Total loss
    total_loss = (pos_loss + neg_loss).sum()

    return total_loss.item()


def main():
    print('\n' + '=' * 60)
    print('TRAINING - MODEL 2 (Skip-gram GPU Optimized)')
    print('=' * 60)
    print('\nConfiguration:')
    print(f"  Embedding dim:    {CONFIG['embedding_dim']}")
    print(f"  Context window:   {CONFIG['context_window']}")
    print(f"  Learning rate:    {CONFIG['learning_rate']}")
    print(f"  Batch size:       {CONFIG['batch_size']}")
    print(f"  Negative samples: {CONFIG['negative_samples']}")
    print(f"  Epochs:           {CONFIG['epochs']}")

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("  Note: Running on CPU. For GPU acceleration, install PyTorch with CUDA")

    start_time = time.time()

    if not os.path.exists(CONFIG['checkpoint_dir']):
        os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

    # ============================================
    # LOAD DICTIONARY
    # ============================================
    print(f"\nLoading dictionary from: {CONFIG['dictionary_file']}")

    input_embeddings_list = []
    output_embeddings_list = []
    word_to_id = {}
    id_to_word = []
    frequencies = []

    emb_dim = CONFIG['embedding_dim']

    with open(CONFIG['dictionary_file'], 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            word = parts[0]
            word_id = int(parts[1])
            frequency = int(parts[2])
            vector_str = parts[3]

            # Parse input embedding
            vector = [float(x) for x in vector_str.split()]
            input_embeddings_list.append(vector)

            # Initialize output embeddings randomly
            import random
            out_vector = [random.uniform(-0.1, 0.1) for _ in range(emb_dim)]
            output_embeddings_list.append(out_vector)

            word_to_id[word] = word_id
            id_to_word.append(word)
            frequencies.append(frequency)

    vocab_size = len(input_embeddings_list)
    print(f"Loaded {vocab_size} words")

    # Convert to tensors
    input_embeddings = torch.tensor(input_embeddings_list, dtype=torch.float32, device=device)
    output_embeddings = torch.tensor(output_embeddings_list, dtype=torch.float32, device=device)

    # Build negative sampling distribution
    neg_sampling_probs = torch.tensor(
        [freq ** 0.75 for freq in frequencies],
        dtype=torch.float32,
        device=device
    )
    neg_sampling_probs /= neg_sampling_probs.sum()

    print('Negative sampling distribution built')

    # ============================================
    # CHECK FOR EXISTING CHECKPOINT
    # ============================================
    start_epoch = 0
    start_file_idx = 0
    pairs_processed = 0
    last_checkpoint_time = time.time()

    progress_file = os.path.join(CONFIG['checkpoint_dir'], 'training_progress.json')

    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)

        start_epoch = progress['epoch']
        start_file_idx = progress.get('fileIdx', 0)
        pairs_processed = progress.get('pairsProcessed', 0)

        print(f"\nFound progress: epoch {start_epoch + 1}, file {start_file_idx}, pairs {pairs_processed}")

        # Load latest checkpoint
        checkpoint_files = [f for f in os.listdir(CONFIG['checkpoint_dir'])
                           if f.startswith('checkpoint_epoch_')]
        if checkpoint_files:
            def extract_number(filename):
                if 'pairs_' in filename:
                    return int(filename.split('pairs_')[1].split('.')[0])
                return 0

            checkpoint_files.sort(key=extract_number, reverse=True)
            latest_checkpoint = checkpoint_files[0]
            checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], latest_checkpoint)

            print(f"Loading checkpoint: {latest_checkpoint}")
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)

            input_embeddings = torch.tensor(checkpoint['embeddings'], dtype=torch.float32, device=device)
            if 'outputEmbeddings' in checkpoint:
                output_embeddings = torch.tensor(checkpoint['outputEmbeddings'], dtype=torch.float32, device=device)

    # ============================================
    # FIND PARQUET FILES
    # ============================================
    print(f"\nScanning parquet files from: {CONFIG['parquet_dir']}")

    all_parquet_files = sorted([f for f in os.listdir(CONFIG['parquet_dir'])
                                if f.endswith('.parquet')])

    max_files = min(CONFIG['max_parquet_files'], len(all_parquet_files))
    parquet_files = all_parquet_files[:max_files]

    print(f"Found {len(all_parquet_files)} parquet files, will process {len(parquet_files)}")

    # ============================================
    # TRAINING LOOP
    # ============================================
    print('\n' + '=' * 60)
    print('STARTING TRAINING')
    print('=' * 60)

    context_size = CONFIG['context_window']
    lr = CONFIG['learning_rate']
    batch_size = CONFIG['batch_size']
    neg_samples = CONFIG['negative_samples']

    for epoch in range(start_epoch, CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        epoch_loss = 0.0
        epoch_pairs = 0

        file_start_idx = start_file_idx if epoch == start_epoch else 0

        for file_idx in range(file_start_idx, len(parquet_files)):
            filename = parquet_files[file_idx]
            filepath = os.path.join(CONFIG['parquet_dir'], filename)

            print(f"\n[File {file_idx + 1}/{len(parquet_files)}] Loading {filename}...")

            # Load ONE parquet file
            df = pd.read_parquet(filepath)
            if 'text' not in df.columns:
                continue

            file_texts = df['text'].tolist()
            print(f"  Loaded {len(file_texts)} documents")

            # Tokenize
            print(f"  Tokenizing...")
            tokenized_docs = []
            for text in file_texts:
                tokens = tokenize_text(text, word_to_id)
                if tokens:
                    tokenized_docs.append(tokens)

            print(f"  Tokenized {len(tokenized_docs)} documents")

            # Free memory
            file_texts = None
            df = None

            # Process pairs inline (streaming) - accumulate into batches
            print(f"  Training on documents (streaming batches)...")
            pair_buffer = []
            batch_count = 0

            for tokens in tokenized_docs:
                num_tokens = len(tokens)
                for center_idx in range(context_size, num_tokens - context_size):
                    center_id = tokens[center_idx]

                    for offset in range(-context_size, context_size + 1):
                        if offset == 0:
                            continue
                        context_id = tokens[center_idx + offset]
                        pair_buffer.append((center_id, context_id))

                        # Train when buffer reaches batch size
                        if len(pair_buffer) >= batch_size:
                            center_ids = torch.tensor([p[0] for p in pair_buffer], dtype=torch.long, device=device)
                            context_ids = torch.tensor([p[1] for p in pair_buffer], dtype=torch.long, device=device)

                            # Train batch on GPU
                            batch_loss = train_batch_gpu(
                                center_ids, context_ids,
                                input_embeddings, output_embeddings,
                                neg_sampling_probs, vocab_size,
                                lr, neg_samples, device
                            )

                            epoch_loss += batch_loss
                            batch_pairs_count = len(pair_buffer)
                            epoch_pairs += batch_pairs_count
                            pairs_processed += batch_pairs_count
                            batch_count += 1

                            # Clear buffer
                            pair_buffer = []

                            # Checkpoint
                            if pairs_processed % CONFIG['checkpoint_every'] < batch_pairs_count:
                                avg_loss = epoch_loss / epoch_pairs
                                current_time = time.time()
                                time_since_last = current_time - last_checkpoint_time
                                pairs_per_second = int(CONFIG['checkpoint_every'] / time_since_last) if time_since_last > 0 else 0

                                print('\n' + '=' * 60)
                                print(f"CHECKPOINT at {pairs_processed} pairs")
                                print('=' * 60)
                                print(f"Epoch: {epoch + 1}/{CONFIG['epochs']}")
                                print(f"File: {file_idx + 1}/{len(parquet_files)}")
                                print(f"Batch: {batch_count}")
                                print(f"Pairs: {pairs_processed:,}")
                                print(f"Avg Loss: {avg_loss:.6f}")
                                print(f"Throughput: {pairs_per_second:,} pairs/s")
                                print(f"Time: {time_since_last:.1f}s")

                                # Save progress
                                progress = {
                                    'epoch': epoch,
                                    'fileIdx': file_idx,
                                    'pairsProcessed': pairs_processed,
                                    'timestamp': int(time.time() * 1000)
                                }
                                with open(progress_file, 'w') as f:
                                    json.dump(progress, f)

                                # Save checkpoint
                                checkpoint = {
                                    'epoch': epoch,
                                    'pairsProcessed': pairs_processed,
                                    'currentFileIdx': file_idx,
                                    'embeddings': input_embeddings.cpu().numpy().tolist(),
                                    'outputEmbeddings': output_embeddings.cpu().numpy().tolist(),
                                    'wordToId': list(word_to_id.items()),
                                    'idToWord': id_to_word,
                                    'vocabSize': vocab_size,
                                    'avgLoss': avg_loss,
                                    'timestamp': int(time.time() * 1000)
                                }

                                checkpoint_path = os.path.join(
                                    CONFIG['checkpoint_dir'],
                                    f"checkpoint_epoch_{epoch}_pairs_{pairs_processed}.json"
                                )

                                with open(checkpoint_path, 'w') as f:
                                    json.dump(checkpoint, f)

                                print(f"Checkpoint saved to: {checkpoint_path}")
                                print('=' * 60 + '\n')

                                last_checkpoint_time = current_time

                            if batch_count % 100 == 0:
                                print(f"    Processed {batch_count} batches ({pairs_processed:,} pairs)...")

            # Process remaining pairs in buffer
            if len(pair_buffer) > 0:
                print(f"  Processing final {len(pair_buffer)} pairs...")

            print(f"  File complete")

        avg_epoch_loss = epoch_loss / epoch_pairs if epoch_pairs > 0 else 0
        print(f"\nEpoch {epoch + 1} complete | Avg Loss: {avg_epoch_loss:.6f} | Pairs: {epoch_pairs:,}")

    # ============================================
    # SAVE FINAL MODEL
    # ============================================
    print('\nSaving final model...')

    final_path = './data/model2_final.ndjson'

    with open(final_path, 'w') as f:
        for i in range(vocab_size):
            word = id_to_word[i]
            vec = input_embeddings[i].cpu().numpy()

            vector_str = ' '.join([f'{v:.4f}' for v in vec])
            line = f"{word}\t{i}\t{vector_str}\n"
            f.write(line)

    elapsed = time.time() - start_time
    print(f"\nTraining complete! Time elapsed: {elapsed:.1f}s")
    print(f"Total pairs processed: {pairs_processed:,}")
    print(f"Average throughput: {pairs_processed/elapsed:.0f} pairs/s")
    print(f"Final model saved to: {final_path}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\nTraining failed: {e}')
        import traceback
        traceback.print_exc()
        exit(1)
