#!/bin/bash
set -exo pipefail

PREDICT_DIR_IDX=1
NUM_PREDICTS=5

MODEL_IDX=1
for SEED in 700 1500; do
    for ((PREDICT_IDX=1; PREDICT_IDX<=NUM_PREDICTS; PREDICT_IDX++)); do
        if [ "$PREDICT_IDX" -eq 1 ] && [ "$SEED" -eq 700 ]; then
            STOCHASTIC_PREDICT=false
        else
            STOCHASTIC_PREDICT=true
        fi

        uv run -m cafa.scripts.cli predict \
            --config src/cafa/experiments/embedding_nn/predict_config.yaml \
            --ckpt_path ${MODELS_DIR}/model_${MODEL_IDX}.ckpt \
            --data.stochastic_predict $STOCHASTIC_PREDICT \
            --data.data_dir ${DATA_DIR}/cafa-6-protein-function-prediction/ \
            --data.train_terms_path ${DATA_DIR}/release_229/train_test_terms.tsv \
            --data.train_propagated_terms_path ${DATA_DIR}/release_229/train_test_propagated_terms.tsv \
            --data.train_seqs_path ${DATA_DIR}/cafa-6-protein-function-prediction/Train/train_sequences.fasta \
            --data.test_seqs_path ${DATA_DIR}/cafa-6-protein-function-prediction/Test/testsuperset.fasta \
            --data.val_dir ${DATA_DIR}/release_229/ \
            --data.terms_evidence_path  ${DATA_DIR}/release_229/test_terms_evidence.tsv \
            --data.predict_proteins_path ${DATA_DIR}/auxiliary/test_protein_ids.csv \
            --data.embedding_paths "[
                ${DATA_DIR}/embeddings/embeddings_esm2_3B.safetensors,
                ${DATA_DIR}/embeddings/embeddings_3Di.safetensors,
                ${DATA_DIR}/embeddings/embeddings_pubmed.safetensors,
            ]" \
            --seed_everything $((SEED + 7*PREDICT_IDX)) \
            --trainer.default_root_dir "${OUTPUT_DIR}/predict_${PREDICT_DIR_IDX}/"
    PREDICT_DIR_IDX=$((PREDICT_DIR_IDX + 1))
    done
done

MODEL_IDX=2
for SEED in 900 1100; do
    for ((PREDICT_IDX=1; PREDICT_IDX<=NUM_PREDICTS; PREDICT_IDX++)); do
        if [ "$PREDICT_IDX" -eq 1 ] && [ "$SEED" -eq 900 ]; then
            STOCHASTIC_PREDICT=false
        else
            STOCHASTIC_PREDICT=true
        fi

        uv run -m cafa.scripts.cli predict \
            --config src/cafa/experiments/embedding_nn/predict_config.yaml \
            --ckpt_path ${MODELS_DIR}/model_${MODEL_IDX}.ckpt \
            --data.stochastic_predict $STOCHASTIC_PREDICT \
            --data.data_dir ${DATA_DIR}/cafa-6-protein-function-prediction/ \
            --data.train_terms_path ${DATA_DIR}/TrainV2/train_test_terms.tsv \
            --data.train_propagated_terms_path ${DATA_DIR}/TrainV2/train_test_propagated_terms.tsv \
            --data.train_seqs_path ${DATA_DIR}/TrainV2/train_sequences.fasta \
            --data.test_seqs_path ${DATA_DIR}/cafa-6-protein-function-prediction/Test/testsuperset.fasta \
            --data.val_dir ${DATA_DIR}/release_229/ \
            --data.terms_evidence_path  ${DATA_DIR}/release_229/test_terms_evidence.tsv \
            --data.predict_proteins_path ${DATA_DIR}/auxiliary/test_protein_ids.csv \
            --data.embedding_paths "[
                ${DATA_DIR}/embeddings/embeddings_esm2_3B_V2.safetensors,
            ]" \
            --seed_everything $((SEED + 7*PREDICT_IDX)) \
            --trainer.default_root_dir "${OUTPUT_DIR}/predict_${PREDICT_DIR_IDX}/"
    PREDICT_DIR_IDX=$((PREDICT_DIR_IDX + 1))
    done
done

MODEL_IDX=3
for SEED in 500 1300; do
    for ((PREDICT_IDX=1; PREDICT_IDX<=NUM_PREDICTS; PREDICT_IDX++)); do
        if [ "$PREDICT_IDX" -eq 1 ] && [ "$SEED" -eq 500 ]; then
            STOCHASTIC_PREDICT=false
        else
            STOCHASTIC_PREDICT=true
        fi

        uv run -m cafa.scripts.cli predict \
            --config src/cafa/experiments/embedding_nn/predict_config.yaml \
            --ckpt_path ${MODELS_DIR}/model_${MODEL_IDX}.ckpt \
            --data.stochastic_predict $STOCHASTIC_PREDICT \
            --data.data_dir ${DATA_DIR}/cafa-6-protein-function-prediction/ \
            --data.train_terms_path ${DATA_DIR}/release_229/train_test_terms.tsv \
            --data.train_propagated_terms_path ${DATA_DIR}/release_229/train_test_propagated_terms.tsv \
            --data.train_seqs_path ${DATA_DIR}/cafa-6-protein-function-prediction/Train/train_sequences.fasta \
            --data.test_seqs_path ${DATA_DIR}/cafa-6-protein-function-prediction/Test/testsuperset.fasta \
            --data.val_dir ${DATA_DIR}/release_229/ \
            --data.terms_evidence_path  ${DATA_DIR}/release_229/test_terms_evidence.tsv \
            --data.predict_proteins_path ${DATA_DIR}/auxiliary/test_protein_ids.csv \
            --data.embedding_paths "[
                ${DATA_DIR}/embeddings/embeddings_esm2_3B.safetensors,
                ${DATA_DIR}/embeddings/embeddings_3Di.safetensors,
                ${DATA_DIR}/embeddings/embeddings_ProstT5.safetensors 
            ]" \
            --seed_everything $((SEED + 7*PREDICT_IDX)) \
            --trainer.default_root_dir "${OUTPUT_DIR}/predict_${PREDICT_DIR_IDX}/"
    PREDICT_DIR_IDX=$((PREDICT_DIR_IDX + 1))
    done
done

MODEL_IDX=4
SEED=300
for ((PREDICT_IDX=1; PREDICT_IDX<=NUM_PREDICTS; PREDICT_IDX++)); do
    if [ "$PREDICT_IDX" -eq 1 ] && [ "$SEED" -eq 300 ]; then
        STOCHASTIC_PREDICT=false
    else
        STOCHASTIC_PREDICT=true
    fi

    uv run -m cafa.scripts.cli predict \
        --config src/cafa/experiments/embedding_nn/predict_config.yaml \
        --ckpt_path ${MODELS_DIR}/model_${MODEL_IDX}.ckpt \
        --data.stochastic_predict $STOCHASTIC_PREDICT \
        --data.data_dir ${DATA_DIR}/cafa-6-protein-function-prediction/ \
        --data.train_terms_path ${DATA_DIR}/release_229/train_test_terms.tsv \
        --data.train_propagated_terms_path ${DATA_DIR}/release_229/train_test_propagated_terms.tsv \
        --data.train_seqs_path ${DATA_DIR}/cafa-6-protein-function-prediction/Train/train_sequences.fasta \
        --data.test_seqs_path ${DATA_DIR}/cafa-6-protein-function-prediction/Test/testsuperset.fasta \
        --data.val_dir ${DATA_DIR}/release_229/ \
        --data.terms_evidence_path  ${DATA_DIR}/release_229/test_terms_evidence.tsv \
        --data.predict_proteins_path ${DATA_DIR}/auxiliary/test_protein_ids.csv \
        --data.embedding_paths "[
            ${DATA_DIR}/embeddings/embeddings_esm2_3B.safetensors,
            ${DATA_DIR}/embeddings/embeddings_3Di.safetensors,
            ${DATA_DIR}/embeddings/embeddings_ProstT5.safetensors,
            ${DATA_DIR}/embeddings/embeddings_pubmed.safetensors
        ]" \
        --seed_everything $((SEED + 7*PREDICT_IDX)) \
        --trainer.default_root_dir "${OUTPUT_DIR}/predict_${PREDICT_DIR_IDX}/"
    PREDICT_DIR_IDX=$((PREDICT_DIR_IDX + 1))
done


prediction_paths=''
for ((PREDICT_IDX=1; PREDICT_IDX<PREDICT_DIR_IDX; PREDICT_IDX++)); do
  prediction_paths+="
     ${OUTPUT_DIR}/predict_${PREDICT_IDX}/predict_epoch_outputs.pkl,"
done

uv run -m cafa.scripts.ensemble_predictions \
    --data_dir ${DATA_DIR}/cafa-6-protein-function-prediction/ \
    --prediction_paths "[
        $prediction_paths
    ]" \
    --output_dir "${OUTPUT_DIR}/ensemble_submission/"
