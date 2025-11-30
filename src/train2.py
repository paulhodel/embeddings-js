"""
Training Script - Model 2 (Skip-gram with Negative Sampling)
Python/CUDA version matching the improved train2.js
- One-file-at-a-time processing (memory efficient)
- Progress tracking with exact resume position
- Checkpointing system
- Windows/second performance metric
"""

import json
import os
import time
import numpy as np
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
    'max_parquet_files': 10,  # Limit number of files to process

    # Architecture
    'embedding_dim': 64,
    'context_window': 3,

    # Training
    'learning_rate': 0.025,
    'batch_size': 20,
    'epochs': 5,
    'negative_samples': 5,

    # Checkpointing
    'checkpoint_every': 10000000,
}


def sigmoid(x):
    """Sigmoid function"""
    return 1.0 / (1.0 + torch.exp(-x))


def tokenize_text(text, word_to_id):
    """
    Fast tokenization matching JavaScript version
    - Lowercase letters
    - Accept letters and numbers, but must contain at least one letter
    - Minimum word length: 2
    """
    tokens = []
    current_word = []
    has_letter = False

    for char in text:
        code = ord(char)

        # Letter (A-Z or a-z)
        if (65 <= code <= 90) or (97 <= code <= 122):
            # Convert to lowercase
            if 65 <= code <= 90:
                current_word.append(chr(code + 32))
            else:
                current_word.append(char)
            has_letter = True
        # Digit (0-9)
        elif 48 <= code <= 57:
            current_word.append(char)
        # Word separator
        else:
            if has_letter and len(current_word) >= 2:
                word = ''.join(current_word)
                word_id = word_to_id.get(word)
                if word_id is not None:
                    tokens.append(word_id)
            current_word = []
            has_letter = False

    # Handle last word
    if has_letter and len(current_word) >= 2:
        word = ''.join(current_word)
        word_id = word_to_id.get(word)
        if word_id is not None:
            tokens.append(word_id)

    return tokens


def main():
    print('\n' + '=' * 60)
    print('TRAINING - MODEL 2 (Skip-gram with CUDA)')
    print('=' * 60)
    print('\nConfiguration:')
    print(f"  Embedding dim:    {CONFIG['embedding_dim']}")
    print(f"  Context window:   {CONFIG['context_window']}")
    print(f"  Learning rate:    {CONFIG['learning_rate']}")
    print(f"  Negative samples: {CONFIG['negative_samples']}")
    print(f"  Epochs:           {CONFIG['epochs']}")

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  Note: Running on CPU. For GPU acceleration, install PyTorch with CUDA")

    start_time = time.time()

    if not os.path.exists(CONFIG['checkpoint_dir']):
        os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

    # ============================================
    # LOAD DICTIONARY
    # ============================================
    print(f"\nLoading dictionary from: {CONFIG['dictionary_file']}")

    input_embeddings = []
    output_embeddings = []
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
            vector = np.array([float(x) for x in vector_str.split()], dtype=np.float32)
            input_embeddings.append(vector)

            # Initialize output embeddings randomly
            out_vector = np.random.uniform(-0.1, 0.1, emb_dim).astype(np.float32)
            output_embeddings.append(out_vector)

            word_to_id[word] = word_id
            id_to_word.append(word)
            frequencies.append(frequency)

    vocab_size = len(input_embeddings)
    print(f"Loaded {vocab_size} words")

    # Convert to tensors
    input_embeddings = torch.tensor(np.array(input_embeddings), dtype=torch.float32, device=device)
    output_embeddings = torch.tensor(np.array(output_embeddings), dtype=torch.float32, device=device)
    frequencies = np.array(frequencies)

    # Build negative sampling distribution (frequency^0.75)
    neg_sampling_probs = np.power(frequencies, 0.75)
    neg_sampling_probs = neg_sampling_probs / neg_sampling_probs.sum()

    # Cumulative distribution for sampling
    cumulative_probs = np.cumsum(neg_sampling_probs)
    cumulative_probs_tensor = torch.tensor(cumulative_probs, dtype=torch.float32, device=device)

    print('Negative sampling distribution built')

    # ============================================
    # CHECK FOR EXISTING PROGRESS
    # ============================================
    start_epoch = 0
    start_file_idx = 0
    start_doc_idx = 0
    batch_count = 0
    last_checkpoint_time = time.time()

    progress_file = os.path.join(CONFIG['checkpoint_dir'], 'training_progress.json')

    if os.path.exists(progress_file):
        print(f"\nFound progress file, loading...")
        with open(progress_file, 'r') as f:
            progress = json.load(f)

        start_epoch = progress['epoch']
        start_file_idx = progress.get('fileIdx', 0)
        start_doc_idx = progress.get('docIdx', 0)
        batch_count = progress['batch']

        print(f"Resuming from epoch {start_epoch}, file {start_file_idx}, doc {start_doc_idx}, batch {batch_count}")

        # Load latest checkpoint to restore embeddings
        checkpoint_files = [f for f in os.listdir(CONFIG['checkpoint_dir'])
                           if f.startswith('checkpoint_epoch_')]
        if checkpoint_files:
            # Sort numerically by batch number
            checkpoint_files.sort(key=lambda x: int(x.split('batch_')[1].split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], latest_checkpoint)

            print(f"Loading checkpoint: {latest_checkpoint}")
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)

            # Restore embeddings
            input_embeddings_list = checkpoint['embeddings']
            for i in range(vocab_size):
                input_embeddings[i] = torch.tensor(input_embeddings_list[i], device=device)

            if 'outputEmbeddings' in checkpoint:
                output_embeddings_list = checkpoint['outputEmbeddings']
                for i in range(vocab_size):
                    output_embeddings[i] = torch.tensor(output_embeddings_list[i], device=device)

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

    # Gradient accumulators
    grad_input = {}
    grad_output = {}
    batch_window_count = 0

    for epoch in range(start_epoch, CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        epoch_loss = 0.0
        windows_processed = 0

        # Process each file
        for file_idx in range(start_file_idx if epoch == start_epoch else 0, len(parquet_files)):
            filename = parquet_files[file_idx]
            filepath = os.path.join(CONFIG['parquet_dir'], filename)

            print(f"\n  [{file_idx + 1}/{len(parquet_files)}] Loading {filename}...")

            # Load ONE parquet file
            df = pd.read_parquet(filepath)
            if 'text' not in df.columns:
                continue

            file_texts = df['text'].tolist()
            print(f"    Loaded {len(file_texts)} documents")

            # Tokenize this file's documents
            print(f"    Tokenizing...")
            tokenized_docs = []
            for text in file_texts:
                tokens = tokenize_text(text, word_to_id)
                if tokens:
                    tokenized_docs.append(tokens)

            print(f"    Tokenized {len(tokenized_docs)} documents")

            # Free memory
            file_texts = None
            df = None

            num_tokenized_docs = len(tokenized_docs)

            # Determine starting document for this file
            doc_start_idx = start_doc_idx if (epoch == start_epoch and file_idx == start_file_idx) else 0

            # Train on this file
            for doc_idx in range(doc_start_idx, num_tokenized_docs):
                tokens = tokenized_docs[doc_idx]
                num_tokens = len(tokens)

                for center in range(context_size, num_tokens - context_size):
                    center_id = tokens[center]
                    center_vec = input_embeddings[center_id]

                    # Process all context words
                    for offset in range(-context_size, context_size + 1):
                        if offset == 0:
                            continue

                        context_id = tokens[center + offset]
                        context_vec = output_embeddings[context_id]

                        # POSITIVE SAMPLE
                        dot = torch.dot(center_vec, context_vec)
                        pred = 1.0 / (1.0 + torch.exp(-dot))
                        epoch_loss += -torch.log(pred + 1e-8).item()
                        grad = pred - 1.0

                        # Accumulate positive gradients
                        if center_id not in grad_input:
                            grad_input[center_id] = torch.zeros(emb_dim, device=device)
                        if context_id not in grad_output:
                            grad_output[context_id] = torch.zeros(emb_dim, device=device)

                        grad_input[center_id] += grad * context_vec
                        grad_output[context_id] += grad * center_vec

                        # NEGATIVE SAMPLES
                        for _ in range(neg_samples):
                            # Sample negative word using cumulative distribution
                            r = torch.rand(1, device=device).item()
                            neg_id = torch.searchsorted(cumulative_probs_tensor, r).item()

                            if neg_id >= vocab_size:
                                neg_id = vocab_size - 1

                            if neg_id == context_id:
                                continue

                            neg_vec = output_embeddings[neg_id]

                            # Negative dot product
                            neg_dot = torch.dot(center_vec, neg_vec)
                            neg_pred = 1.0 / (1.0 + torch.exp(-neg_dot))
                            epoch_loss += -torch.log(1.0 - neg_pred + 1e-8).item()
                            neg_grad = neg_pred

                            # Accumulate negative gradients
                            if neg_id not in grad_output:
                                grad_output[neg_id] = torch.zeros(emb_dim, device=device)

                            grad_input[center_id] += neg_grad * neg_vec
                            grad_output[neg_id] += neg_grad * center_vec

                        windows_processed += 1
                        batch_window_count += 1
                        batch_count += 1

                        # Apply batch update
                        if batch_window_count == batch_size:
                            with torch.no_grad():
                                # Update input embeddings and normalize
                                for emb_id, grad in grad_input.items():
                                    vec = input_embeddings[emb_id]
                                    vec -= lr * (grad / batch_size)

                                    # L2 normalize
                                    norm = torch.norm(vec) + 1e-8
                                    vec /= norm

                                # Update output embeddings
                                for emb_id, grad in grad_output.items():
                                    vec = output_embeddings[emb_id]
                                    vec -= lr * (grad / batch_size)

                            grad_input.clear()
                            grad_output.clear()
                            batch_window_count = 0

                        # Checkpoint
                        if batch_count % CONFIG['checkpoint_every'] == 0:
                            avg_loss = epoch_loss / windows_processed if windows_processed > 0 else 0
                            current_time = time.time()
                            time_since_last = current_time - last_checkpoint_time
                            windows_per_second = int(CONFIG['checkpoint_every'] / time_since_last) if time_since_last > 0 else 0

                            print('\n' + '=' * 60)
                            print(f"CHECKPOINT at batch {batch_count}")
                            print('=' * 60)
                            print(f"Epoch: {epoch + 1}/{CONFIG['epochs']}")
                            print(f"File: {file_idx + 1}/{len(parquet_files)} - {filename}")
                            print(f"Document: {doc_idx + 1}/{num_tokenized_docs}")
                            print(f"Windows: {windows_processed}")
                            print(f"Avg Loss: {avg_loss:.6f}")
                            print(f"Time for last {CONFIG['checkpoint_every']} batches: {time_since_last:.1f}s ({windows_per_second} windows/s)")

                            # Save progress file (lightweight)
                            progress = {
                                'epoch': epoch,
                                'fileIdx': file_idx,
                                'docIdx': doc_idx,
                                'batch': batch_count,
                                'timestamp': int(time.time() * 1000)
                            }
                            with open(progress_file, 'w') as f:
                                json.dump(progress, f)

                            # Save full checkpoint (heavy)
                            checkpoint = {
                                'epoch': epoch,
                                'batch': batch_count,
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
                                f"checkpoint_epoch_{epoch}_batch_{batch_count}.json"
                            )

                            with open(checkpoint_path, 'w') as f:
                                json.dump(checkpoint, f)

                            print(f"Checkpoint saved to: {checkpoint_path}")
                            print('=' * 60 + '\n')

                            last_checkpoint_time = current_time

                if (doc_idx + 1) % 5000 == 0:
                    print(f"      Processed {doc_idx + 1}/{num_tokenized_docs} documents")

            # Reset start_doc_idx after first file in resumed epoch
            if epoch == start_epoch and file_idx == start_file_idx:
                start_doc_idx = 0

            # Free memory
            tokenized_docs = None

        # Reset start_file_idx after first epoch
        if epoch == start_epoch:
            start_file_idx = 0

        # Apply remaining gradients at end of epoch
        if batch_window_count > 0:
            with torch.no_grad():
                for emb_id, grad in grad_input.items():
                    vec = input_embeddings[emb_id]
                    vec -= lr * (grad / batch_window_count)

                    norm = torch.norm(vec) + 1e-8
                    vec /= norm

                for emb_id, grad in grad_output.items():
                    vec = output_embeddings[emb_id]
                    vec -= lr * (grad / batch_window_count)

            grad_input.clear()
            grad_output.clear()
            batch_window_count = 0

        avg_epoch_loss = epoch_loss / windows_processed if windows_processed > 0 else 0
        print(f"\nEpoch {epoch + 1} complete | Avg Loss: {avg_epoch_loss:.6f}")

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

    elapsed = (time.time() - start_time)
    print(f"\nTraining complete! Time elapsed: {elapsed:.1f}s")
    print(f"Final model saved to: {final_path}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\n❌ Training failed: {e}')
        import traceback
        traceback.print_exc()
        exit(1)
