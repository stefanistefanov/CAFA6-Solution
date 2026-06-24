import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import safetensors.torch
import torch
from sklearn.calibration import LabelEncoder
from torch.utils.data import default_collate

from cafa.core.constants import UNKNOWN_INPUT_TERM, UNKNOWN_INPUT_TERM_IDX
from cafa.core.datamodule import ProteinDataModule
from cafa.experiments.embedding_nn.dataset import EmbeddingDataset
from cafa.core.common.utils import load_pickle

logger = logging.getLogger(__name__)


class ProteinEmbeddingDataModule(ProteinDataModule):
    def __init__(
        self,
        data_dir: str,
        train_terms_path: str,
        train_propagated_terms_path: str,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        fold_path: str | None,
        fold: int | None,
        train_seqs_path: str | None,
        test_seqs_path: str | None,
        embedding_paths: list[str],
        checkpoint_path: str | None = None,
        terms_evidence_path: str | None = None,
        predict_proteins_path: str | None = None,
        subontology: str = None,
        val_dir: str = None,
        stochastic_predict: bool = False,
        use_cafa6_metrics: bool = False,
    ):
        super().__init__(
            data_dir=data_dir,
            train_terms_path=train_terms_path,
            train_propagated_terms_path=train_propagated_terms_path,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            fold_path=fold_path,
            fold=fold,
            train_seqs_path=train_seqs_path,
            test_seqs_path=test_seqs_path,
            predict_proteins_path=predict_proteins_path,
            subontology=subontology,
            val_dir=val_dir,
            stochastic_predict=stochastic_predict,
            use_cafa6_metrics=use_cafa6_metrics,
        )
        self.save_hyperparameters()
        self.dataset_columns += [
            'embedding',
            'taxon_lineage_indices',
            'norm_seq_len',
            'original_term_indices',
            'term_indices',
            'evidence_indices',
        ]
        self.embedding_dims = None

    def get_train_dataset(self, examples: List[Dict[str, Any]]):
        return EmbeddingDataset(
            examples,
            train=True,
            stochastic_predict=False,
            embedding_dims=self.embedding_dims,
            subontology_ranges=self.subontology_ranges,
        )

    def get_val_dataset(self, examples: List[Dict[str, Any]]):
        return EmbeddingDataset(
            examples,
            train=False,
            stochastic_predict=self.hparams.stochastic_predict,
            embedding_dims=self.embedding_dims,
            subontology_ranges=self.subontology_ranges,
        )


    def populate_examples_data(self, examples_df: pd.DataFrame):
        super().populate_examples_data(examples_df)

        protein_ids = set(examples_df['protein_id'])
        self.embedding_dims = []
        embedding_by_protein_id = None
        for embedding_path in self.hparams.embedding_paths:
            single_embedding_by_protein_id = safetensors.torch.load_file(
                embedding_path
            )
            first_embedding = next(
                iter(single_embedding_by_protein_id.values())
            )
            embedding_protein_ids = set(single_embedding_by_protein_id.keys())
            missing_embedding_protein_ids = protein_ids - embedding_protein_ids
            logger.info(
                f'{len(missing_embedding_protein_ids)=} for {embedding_path}'
            )
            single_embedding_by_protein_id.update(
                {
                    missing_protein_id: torch.zeros_like(first_embedding)
                    for missing_protein_id in missing_embedding_protein_ids
                }
            )

            self.embedding_dims.append(first_embedding.shape[0])
            if embedding_by_protein_id is None:
                embedding_by_protein_id = single_embedding_by_protein_id
            else:
                for (
                    protein_id,
                    embedding,
                ) in single_embedding_by_protein_id.items():
                    embedding_by_protein_id[protein_id] = torch.cat(
                        (
                            embedding_by_protein_id[protein_id],
                            embedding,
                        ),
                        dim=0,
                    )
            del single_embedding_by_protein_id
        logger.info(f'{self.embedding_dims=}')

        examples_df['embedding'] = examples_df['protein_id'].map(
            embedding_by_protein_id
        )

        if self.hparams.terms_evidence_path is not None:
            terms_evidence_df = pd.read_csv(
                self.hparams.terms_evidence_path, sep='\t'
            )
            logger.info(
                f'terms_evidence_df unique terms {terms_evidence_df["term"].nunique()}'
            )
            terms_evidence_df = terms_evidence_df.groupby(
                'EntryID', sort=False
            ).agg(
                input_terms=('term', list),
                input_evidences=('Evidence', list),
            )
            terms_evidence_df = terms_evidence_df.rename(
                index={'EntryID': 'protein_id'}
            )

            examples_df.set_index('protein_id', inplace=True)
            examples_df['input_terms'] = terms_evidence_df['input_terms']
            examples_df['input_evidences'] = terms_evidence_df[
                'input_evidences'
            ]
            examples_df['input_terms'] = examples_df['input_terms'].apply(
                lambda x: x if isinstance(x, list) else []
            )
            examples_df['input_evidences'] = examples_df[
                'input_evidences'
            ].apply(lambda x: x if isinstance(x, list) else [])
            examples_df.reset_index(inplace=True)

    def fit_preprocessors(self, train_examples_df: pd.DataFrame):
        super().fit_preprocessors(train_examples_df)

        # The same order of terms as in multi_label_binarizer is required
        # to have consistent term indices when adding prediction logits
        self.term_encoder = LabelEncoder()
        self.term_encoder.classes_ = np.insert(
            self.multi_label_binarizer.classes_,
            0,
            UNKNOWN_INPUT_TERM,
        )

        self.evidence_encoder = LabelEncoder()
        # Non-experimental evidence codes only
        self.evidence_encoder.fit(
            y=[
                'IEA',
                'IBA',
                'ISS',
                'ISO',
                'NAS',
                'ND',
                'ISA',
                'ISM',
                'RCA',
                'IGC',
            ]
        )

        if self.hparams.checkpoint_path is not None:
            logger.info(
                'Loading datamodule state from checkpoint'
                f' {self.hparams.checkpoint_path}'
            )
            state_dict = torch.load(
                self.hparams.checkpoint_path,
                weights_only=False,
                map_location='cpu',
            )['ProteinEmbeddingDataModule']
            self.train_input_term_indices = state_dict.get(
                'train_input_term_indices', set()
            )
            del state_dict
        else:
            train_input_terms = sorted(
                {
                    term
                    for input_terms in train_examples_df[
                        'input_terms'
                    ].to_list()
                    for term in input_terms
                }
            )
            self.train_input_term_indices = set(
                self.term_encoder.transform(train_input_terms)
            )
        logger.info(f'fit_preprocessors {len(self.train_input_term_indices)=}')

        self.input_terms_size = len(self.term_encoder.classes_)
        self.input_evidences_size = len(self.evidence_encoder.classes_)
        logger.info(f'{self.input_terms_size=} {self.input_evidences_size=}')

    def state_dict(self):
        datamodule_state_dict = super().state_dict()
        datamodule_state_dict['train_input_term_indices'] = (
            self.train_input_term_indices
        )
        return datamodule_state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # Skipping loading state_dict, the supported state should be already
        # loaded at setup
        logger.info('Skipping loading state_dict in datamodule')

    def preprocess_examples_data(self, examples_df: pd.DataFrame):
        super().preprocess_examples_data(examples_df)
        term_encoder_map = {
            term: term_idx
            for term_idx, term in enumerate(self.term_encoder.classes_.tolist())
        }
        logger.info(
            f'preprocess_examples_data {len(self.train_input_term_indices)=}'
        )
        assert term_encoder_map[UNKNOWN_INPUT_TERM] == UNKNOWN_INPUT_TERM_IDX
        
        examples_df['original_term_indices'] = [
            [term_encoder_map[term] for term in input_terms]
            for input_terms in examples_df['input_terms']
        ]
        examples_df['term_indices'] = [
            [
                (
                    term_index
                    if term_index in self.train_input_term_indices
                    else UNKNOWN_INPUT_TERM_IDX
                )
                for term_index in term_indices
            ]
            for term_indices in examples_df['original_term_indices']
        ]

        evidence_encoder_map = {
            evidence: evidence_idx
            for evidence_idx, evidence in enumerate(
                self.evidence_encoder.classes_.tolist()
            )
        }
        examples_df['evidence_indices'] = [
            [evidence_encoder_map[evidence] for evidence in input_evidences]
            for input_evidences in examples_df['input_evidences']
        ]

    def get_collate_fn(self):
        return self.collate

    def collate(self, batch: List[Dict[str, Any]]):
        max_input_terms = max(item['num_input_terms'] for item in batch)
        max_num_taxons = max(
            item['taxon_lineage_indices'].shape[0] for item in batch
        )
        batch_taxon_indices = []
        batch_original_term_indices = []
        batch_term_indices = []
        batch_evidence_indices = []
        for item in batch:
            item_taxon_indices = item.pop('taxon_lineage_indices')
            item_num_taxons = item_taxon_indices.shape[0]
            padded_taxon_indices = torch.full(
                (max_num_taxons,), fill_value=-1, dtype=torch.long
            )
            padded_taxon_indices[:item_num_taxons] = item_taxon_indices
            batch_taxon_indices.append(padded_taxon_indices)

            item_original_term_indices = item.pop('original_term_indices')
            item_term_indices = item.pop('term_indices')
            item_evidence_indices = item.pop('evidence_indices')
            num_input_terms = item['num_input_terms']

            padded_original_terms = torch.full(
                (max_input_terms,), fill_value=-1, dtype=torch.long
            )
            padded_terms = torch.full(
                (max_input_terms,), fill_value=-1, dtype=torch.long
            )
            padded_evidence = torch.full(
                (max_input_terms,), fill_value=-1, dtype=torch.long
            )

            padded_original_terms[:num_input_terms] = item_original_term_indices
            padded_terms[:num_input_terms] = item_term_indices
            padded_evidence[:num_input_terms] = item_evidence_indices

            batch_original_term_indices.append(padded_original_terms)
            batch_term_indices.append(padded_terms)
            batch_evidence_indices.append(padded_evidence)
        collated_padded_batch = {
            'taxon_lineage_indices': torch.stack(
                batch_taxon_indices
            ),  # (batch_size, max_num_taxons)
            'original_term_indices': torch.stack(
                batch_original_term_indices
            ),  # (batch_size, max_input_terms)
            'term_indices': torch.stack(
                batch_term_indices
            ),  # (batch_size, max_input_terms)
            'evidence_indices': torch.stack(
                batch_evidence_indices
            ),  # (batch_size, max_input_terms)
        }

        collated_batch = default_collate(batch)
        collated_batch.update(**collated_padded_batch)

        return collated_batch
