import ijson
import pandas as pd
import os
from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def parse_json_shard(file_path):
    """
    Parses a single JSON shard using ijson for memory efficiency.
    Yields paper dictionaries.
    """
    # TODO: Implement streaming parsing logic
    # with open(file_path, 'r') as f:
    #     objects = ijson.items(f, 'item')
    #     for obj in objects:
    #         yield obj
    pass

def ingest_data():
    """
    Main function to ingest all JSON shards from RAW_DATA_DIR.
    Should handle chunking and initial cleaning.
    """
    print(f"Scanning {RAW_DATA_DIR} for JSON shards...")
    # TODO: Iterate over dblp-ref-*.json files
    # TODO: Process in chunks and save to intermediate format or directly to Parquet
    pass

if __name__ == "__main__":
    ingest_data()
