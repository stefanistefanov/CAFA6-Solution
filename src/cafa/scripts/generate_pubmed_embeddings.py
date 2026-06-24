import logging
from pathlib import Path

import torch
from datasets import load_from_disk
from jsonargparse import CLI
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def generate_pubmed_embeddings(
    dataset_dir: str,
    output_dir: str,
    model_name: str = "NeuML/pubmedbert-base-embeddings",
    batch_size: int = 32,
    max_length: int = 512,
    device: str = None,
    embeddings_per_shard: int = 300000,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    dataset = load_from_disk(dataset_dir)
    logger.info(f"Dataset loaded with {len(dataset)} records")

    if "PMID" not in dataset.column_names or "Text" not in dataset.column_names:
        raise ValueError(
            f"Dataset must have 'PMID' and 'Text' fields. Found: {dataset.column_names}"
        )

    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_length
    logger.info(f"Model loaded. Max sequence length: {model.max_seq_length}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_records = len(dataset)
    num_shards = (
        total_records + embeddings_per_shard - 1
    ) // embeddings_per_shard
    logger.info(
        f"Will save {total_records} embeddings across {num_shards} shards"
    )

    pmid_to_embedding = {}
    shard_idx = 1
    total_saved = 0

    logger.info(f"Generating embeddings with batch size {batch_size}")

    for i in tqdm(
        range(0, len(dataset), batch_size), desc="Processing batches"
    ):
        batch = dataset[i : i + batch_size]
        texts = batch["Text"]
        pmids = batch["PMID"]

        embeddings = model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=device,
            batch_size=batch_size,
        )

        for pmid, embedding in zip(pmids, embeddings):
            pmid_to_embedding[str(pmid)] = embedding.cpu()

        if len(pmid_to_embedding) >= embeddings_per_shard:
            shard_filename = (
                f"embeddings-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
            )
            shard_path = output_path / shard_filename
            logger.info(
                f"Saving shard {shard_idx}/{num_shards} with {len(pmid_to_embedding)} embeddings to {shard_filename}"
            )
            save_file(pmid_to_embedding, str(shard_path))
            total_saved += len(pmid_to_embedding)
            shard_idx += 1
            pmid_to_embedding = {}  # Clear memory

    # Save remaining embeddings
    if len(pmid_to_embedding) > 0:
        shard_filename = (
            f"embeddings-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
        )
        shard_path = output_path / shard_filename
        logger.info(
            f"Saving final shard {shard_idx}/{num_shards} with {len(pmid_to_embedding)} embeddings to {shard_filename}"
        )
        save_file(pmid_to_embedding, str(shard_path))
        total_saved += len(pmid_to_embedding)

    logger.info(
        f"Total embeddings saved: {total_saved} across {num_shards} shards"
    )

if __name__ == "__main__":
    CLI(generate_pubmed_embeddings, as_positional=False)
