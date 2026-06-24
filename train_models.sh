#!/bin/bash
set -exo pipefail

uv run -m cafa.scripts.cli fit \
    --config src/cafa/experiments/embedding_nn/fit_config.yaml \
    --seed_everything 703 \
    --data.data_dir  ${DATA_DIR}/cafa-6-protein-function-prediction/ \
    --data.train_terms_path ${DATA_DIR}/release_229/train_test_terms.tsv \
    --data.train_propagated_terms_path ${DATA_DIR}/release_229/train_test_propagated_terms.tsv \
    --data.train_seqs_path ${DATA_DIR}/cafa-6-protein-function-prediction/Train/train_sequences.fasta \
    --data.test_seqs_path ${DATA_DIR}/cafa-6-protein-function-prediction/Test/testsuperset.fasta \
    --data.val_dir ${DATA_DIR}/release_229/ \
    --data.terms_evidence_path  ${DATA_DIR}/release_229/test_terms_evidence.tsv \
    --data.embedding_paths "[
        ${DATA_DIR}/embeddings/embeddings_esm2_3B.safetensors,
        ${DATA_DIR}/embeddings/embeddings_3Di.safetensors,
        ${DATA_DIR}/embeddings/embeddings_pubmed.safetensors,
    ]" \
    --trainer.default_root_dir $TRAIN_OUTPUT_DIR/model_1/

uv run -m cafa.scripts.cli fit \
    --config src/cafa/experiments/embedding_nn/fit_config.yaml \
    --seed_everything 901 \
    --data.data_dir ${DATA_DIR}/cafa-6-protein-function-prediction/ \
    --data.train_terms_path ${DATA_DIR}/TrainV2/train_test_terms.tsv \
    --data.train_propagated_terms_path ${DATA_DIR}/TrainV2/train_test_propagated_terms.tsv \
    --data.train_seqs_path ${DATA_DIR}/TrainV2/train_sequences.fasta \
    --data.test_seqs_path ${DATA_DIR}/cafa-6-protein-function-prediction/Test/testsuperset.fasta \
    --data.val_dir ${DATA_DIR}/release_229/ \
    --data.terms_evidence_path  ${DATA_DIR}/TrainV2/test_terms_evidence.tsv \
    --data.embedding_paths "[
        ${DATA_DIR}/embeddings/embeddings_esm2_3B_V2.safetensors,
    ]" \
   --trainer.default_root_dir $TRAIN_OUTPUT_DIR/model_2/

uv run -m cafa.scripts.cli fit \
    --config src/cafa/experiments/embedding_nn/fit_config.yaml \
    --seed_everything 501 \
    --data.data_dir ${DATA_DIR}/cafa-6-protein-function-prediction/ \
    --data.train_terms_path ${DATA_DIR}/release_229/train_test_terms.tsv \
    --data.train_propagated_terms_path ${DATA_DIR}/release_229/train_test_propagated_terms.tsv \
    --data.train_seqs_path ${DATA_DIR}/cafa-6-protein-function-prediction/Train/train_sequences.fasta \
    --data.test_seqs_path ${DATA_DIR}/cafa-6-protein-function-prediction/Test/testsuperset.fasta \
    --data.val_dir ${DATA_DIR}/release_229/ \
    --data.terms_evidence_path ${DATA_DIR}/release_229/test_terms_evidence.tsv \
    --data.embedding_paths "[
        ${DATA_DIR}/embeddings/embeddings_esm2_3B.safetensors,
        ${DATA_DIR}/embeddings/embeddings_3Di.safetensors,
        ${DATA_DIR}/embeddings/embeddings_ProstT5.safetensors 
    ]" \
   --trainer.default_root_dir $TRAIN_OUTPUT_DIR/model_3/

uv run -m cafa.scripts.cli fit \
    --config src/cafa/experiments/embedding_nn/fit_config.yaml \
    --seed_everything 301 \
    --data.data_dir ${DATA_DIR}/cafa-6-protein-function-prediction/ \
    --data.train_terms_path ${DATA_DIR}/release_229/train_test_terms.tsv \
    --data.train_propagated_terms_path ${DATA_DIR}/release_229/train_test_propagated_terms.tsv \
    --data.train_seqs_path ${DATA_DIR}/cafa-6-protein-function-prediction/Train/train_sequences.fasta \
    --data.test_seqs_path ${DATA_DIR}/cafa-6-protein-function-prediction/Test/testsuperset.fasta \
    --data.val_dir ${DATA_DIR}/release_229/ \
    --data.terms_evidence_path ${DATA_DIR}/release_229/test_terms_evidence.tsv \
    --data.embedding_paths "[
        ${DATA_DIR}/embeddings/embeddings_esm2_3B.safetensors,
        ${DATA_DIR}/embeddings/embeddings_3Di.safetensors,
        ${DATA_DIR}/embeddings/embeddings_ProstT5.safetensors,
        ${DATA_DIR}/embeddings/embeddings_pubmed.safetensors
    ]" \
   --trainer.default_root_dir $TRAIN_OUTPUT_DIR/model_4/
