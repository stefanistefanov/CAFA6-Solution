# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        unzip \
        python3 \
        python3-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

RUN pip install --break-system-packages kaggle==1.7.4.5

ENV DATA_DIR=/data
ENV MODELS_DIR=/models
ENV DATASET_3DI_DIR=/data_3di
ENV PUBMED_DATASET_DIR=/data_pubmed
ENV OUTPUT_DIR=/output
ENV TRAIN_OUTPUT_DIR=${OUTPUT_DIR}
ENV EMBEDDINGS_OUTPUT_DIR=${OUTPUT_DIR}
RUN mkdir -p ${DATA_DIR} ${MODELS_DIR} ${DATASET_3DI_DIR} ${PUBMED_DATASET_DIR} ${OUTPUT_DIR}

RUN --mount=type=secret,id=kaggle,target=/root/.kaggle/kaggle.json \
    kaggle datasets download stefanstefanov/cafa6-data --unzip -p ${DATA_DIR} && \
    # Competition dataset, placed inside DATA_DIR/cafa-6-protein-function-prediction \
    kaggle competitions download -c cafa-6-protein-function-prediction -p ${DATA_DIR} && \
    unzip -q ${DATA_DIR}/cafa-6-protein-function-prediction.zip \
        -d ${DATA_DIR}/cafa-6-protein-function-prediction && \
    rm -f ${DATA_DIR}/cafa-6-protein-function-prediction.zip && \
    kaggle datasets download stefanstefanov/cafa6-models --unzip -p ${MODELS_DIR} && \
    # Datasets, only needed for generating embeddings with create_embeddings.sh \
    kaggle datasets download stefanstefanov/cafa6-3di --unzip -p ${DATASET_3DI_DIR} && \
    kaggle datasets download stefanstefanov/cafa6-pubmed --unzip -p ${PUBMED_DATASET_DIR}

WORKDIR /opt
RUN git clone https://github.com/stefanistefanov/CAFA6-Solution.git cafa6-solution
WORKDIR /opt/cafa6-solution

RUN uv sync && uv pip install -e .

RUN cd /data/auxiliary/taxdump && tar -czvf ../taxdump.tar.gz * && cd -

ENTRYPOINT ["./predict_ensemble.sh"]
