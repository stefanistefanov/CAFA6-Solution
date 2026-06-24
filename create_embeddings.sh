#!/bin/bash
set -exo pipefail

uv run -m cafa.scripts.cli predict \
    --config src/cafa/experiments/sequence_embedder/esm2_config.yaml \
    --data.data_dir ${DATA_DIR}/cafa-6-protein-function-prediction/ \
    --data.train_terms_path ${DATA_DIR}/release_229/train_test_terms.tsv \
    --data.train_propagated_terms_path ${DATA_DIR}/release_229/train_test_propagated_terms.tsv \
    --data.model_name facebook/esm2_t36_3B_UR50D \
    --data.test_seqs_path  ${DATA_DIR}/cafa-6-protein-function-prediction/Test/testsuperset.fasta \
    --data.predict_proteins_path ${DATA_DIR}/auxiliary/test_protein_ids.csv \
    --data.fold_path ${DATA_DIR}/auxiliary/fold.csv \
    --trainer.default_root_dir ${EMBEDDINGS_OUTPUT_DIR}/embeddings_esm2_3B

uv run -m cafa.scripts.cli predict \
    --config src/cafa/experiments/sequence_embedder/prost_t5_config.yaml \
    --data.data_dir ${DATA_DIR}/cafa-6-protein-function-prediction/ \
    --data.train_terms_path ${DATA_DIR}/release_229/train_test_terms.tsv \
    --data.train_propagated_terms_path ${DATA_DIR}/release_229/train_test_propagated_terms.tsv \
    --data.test_seqs_path  ${DATA_DIR}/cafa-6-protein-function-prediction/Test/testsuperset.fasta \
    --data.predict_proteins_path ${DATA_DIR}/auxiliary/test_protein_ids.csv \
    --data.fold_path ${DATA_DIR}/auxiliary/fold.csv \
    --trainer.default_root_dir ${EMBEDDINGS_OUTPUT_DIR}/embeddings_ProstT5

uv run -m cafa.scripts.cli predict \
    --config src/cafa/experiments/sequence_embedder/prost_t5_config.yaml \
    --data.data_dir ${DATA_DIR}/cafa-6-protein-function-prediction/ \
    --data.train_terms_path ${DATA_DIR}/release_229/train_test_terms.tsv \
    --data.train_propagated_terms_path ${DATA_DIR}/release_229/train_test_propagated_terms.tsv \
    --data.max_sequence_len 7168 \
    --data.test_seqs_path ${DATASET_3DI_DIR}/test_sequences_3Di.fasta \
    --data.predict_proteins_path ${DATA_DIR}/auxiliary/test_protein_ids.csv \
    --data.fold_path ${DATA_DIR}/auxiliary/fold.csv \
    --trainer.default_root_dir ${EMBEDDINGS_OUTPUT_DIR}/embeddings_3Di


uv run -m cafa.scripts.generate_pubmed_embeddings \
    --dataset_dir ${PUBMED_DATASET_DIR} \
    --batch_size 256 \
    --output_dir ${EMBEDDINGS_OUTPUT_DIR}/embeddings_pubmed_generate

# Use mean and max of the top 5 most recent PubMed embeddings for each protein
uv run -m cafa.scripts.create_protein_pubmed_embeddings \
    --top_k 5 \
    --idmapping_path ${PUBMED_DATASET_DIR}/test_idmapping.tsv \
    --dataset_dir ${PUBMED_DATASET_DIR} \
    --pubmed_embeddings_dir ${EMBEDDINGS_OUTPUT_DIR}/embeddings_pubmed_generate \
    --output_embeddings_path ${EMBEDDINGS_OUTPUT_DIR}/embeddings_pubmed/embeddings_pubmed.safetensors
