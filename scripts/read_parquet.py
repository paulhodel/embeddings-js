"""
Read Parquet files and output as JSON
Used by prepare_data.js to process Parquet files
"""

import sys
import json
import pandas as pd
from pathlib import Path

def read_parquet_file(filepath, max_docs=None):
    """Read a Parquet file and return text column as list"""
    try:
        df = pd.read_parquet(filepath)

        if 'text' not in df.columns:
            print(f"Error: 'text' column not found. Available columns: {df.columns.tolist()}", file=sys.stderr)
            return []

        texts = df['text'].tolist()

        if max_docs:
            texts = texts[:max_docs]

        return texts
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return []

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python read_parquet.py <parquet_file> [max_docs]", file=sys.stderr)
        sys.exit(1)

    filepath = sys.argv[1]
    max_docs = int(sys.argv[2]) if len(sys.argv) > 2 else None

    texts = read_parquet_file(filepath, max_docs)

    # Output as JSON to stdout
    json.dump(texts, sys.stdout)
