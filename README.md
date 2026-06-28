## 6th Place Solution for the CAFA 6 Protein Function Prediction

Instructions to reproduce the solution are as follows.

### Hardware Requirements
The solution was generated on a machine (Ubuntu 24.04 OS) with the following specifications:
- 1 GPU RTX 3090 with 24GB VRAM
- 64 GB RAM
- 12 CPUs
- 200 GB free disk space

### Environment Setup
In the repository root run the following commands to create a virtual environment with the necessary requirements
```bash
uv sync
uv pip install -e .
```

### Datasets Download
The competition dataset and other datasets needed for the solution are available on Kaggle and can be downloaded from there.
```bash
export DATA_DIR=<DATA_DIR>
kaggle datasets download stefanstefanov/cafa6-data --unzip -p $DATA_DIR

# taxdump.tar.gz archive is needed for ete3.NCBITaxa
cd ${DATA_DIR}/auxiliary/taxdump && tar -czvf ../taxdump.tar.gz * && cd -

# The competition dataset should be placed inside the $DATA_DIR in `cafa-6-protein-function-prediction` directory
kaggle competitions download -c cafa-6-protein-function-prediction -p $DATA_DIR
unzip $DATA_DIR/cafa-6-protein-function-prediction.zip -d $DATA_DIR/cafa-6-protein-function-prediction

# Already trained models for the solution are available in the cafa6-models dataset
export MODELS_DIR=<MODELS_DIR>
kaggle datasets download stefanstefanov/cafa6-models --unzip -p $MODELS_DIR

# cafa6-3di and cafa6-pubmed datasets are optional and only needed for generating the embeddings with the `create_embeddings.sh` script.
# The already generated embeddings are included in the cafa6-data dataset and they can be used directly for training and prediction.
export DATASET_3DI_DIR=<DATASET_3DI_DIR>
kaggle datasets download stefanstefanov/cafa6-3di --unzip -p $DATASET_3DI_DIR

export PUBMED_DATASET_DIR=<PUBMED_DATASET_DIR>
kaggle datasets download stefanstefanov/cafa6-pubmed --unzip -p $PUBMED_DATASET_DIR
```

### Predict and Generate Ensemble Submission
The script with commands for generating the predictions and the ensemble submission is `predict_ensemble.sh`. DATA_DIR, MODELS_DIR, and OUTPUT_DIR should be set before running the script. The script will generate 35 predictions with different random seeds and data augmentations enabled during prediction. The final submission is an average of these 35 predictions. Each prediction takes ~20 mins and the time for generating the ensemble submission will be ~12 hours.
```bash
export OUTPUT_DIR=<OUTPUT_DIR>
./predict_ensemble.sh
```

### Models Training
The script with commands for training the models is `train_models.sh`. DATA_DIR and TRAIN_OUTPUT_DIR should be set before running the script. The script will train 4 models with different inputs. Training time for a single model is ~1 hour and the time for training all 4 models will be ~4 hours.
```bash
export TRAIN_OUTPUT_DIR=<TRAIN_OUTPUT_DIR>
./train_models.sh
```

### Generate Embeddings
The script with commands for generating the embeddings is `create_embeddings.sh`. DATA_DIR, DATASET_3DI_DIR, PUBMED_DATASET_DIR and EMBEDDINGS_OUTPUT_DIR should be set before running the script. Time for generating single embeddings varies depending on the model, sequence lengths, dataset size etc. with indicative time of ~7 hours for esm2_t36_3B_UR50D embeddings. So the time for generating all 5 embeddings can be significant 1-2 days.
```bash
export EMBEDDINGS_OUTPUT_DIR=<EMBEDDINGS_OUTPUT_DIR>
./create_embeddings.sh
```

### Docker Container
The solution can also be run inside a Docker container. The Dockerfile is provided in the repository.
```bash
# The kaggle secret is needed for the competition dataset download.
# It is mounted only at build time and not baked into the image.
docker build --secret id=kaggle,src=$HOME/.kaggle/kaggle.json -t cafa6-solution .

# OUTPUT_DIR is an empty directory for storing output artifacts
docker run --gpus device=0  --shm-size=2g -v ${OUTPUT_DIR}:/output cafa6-solution
```
