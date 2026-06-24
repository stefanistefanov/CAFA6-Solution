import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from ete3 import NCBITaxa
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.states import TrainerFn
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset

from cafa.core.constants import UNKNOWN_TAXON_ID
from cafa.core.eval_metrics import EvalMetrics
from cafa.core.metrics import CAFAMetrics, Subontology
from cafa.core.utils import extract_protein_id_from_fasta

logger = logging.getLogger(__name__)


class ProteinDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_terms_path: str,
        train_propagated_terms_path: str,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        fold_path: str| None,
        fold: int | None,
        train_seqs_path: str | None,
        test_seqs_path: str | None,
        predict_proteins_path: str = None,
        subontology: str = None,
        val_dir: str = None,
        stochastic_predict: bool = False,
        use_cafa6_metrics: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.subontology = (
            Subontology[subontology] if subontology is not None else None
        )
        self.multi_label_binarizer = None
        self.num_labels = None
        self.dataset_columns = ['protein_id']

    def setup(self, stage=None):
        logger.info(f'{stage=}')
        is_predict = stage == TrainerFn.PREDICTING

        self.random_gen = np.random.default_rng(
            int(os.environ.get('PL_GLOBAL_SEED'))
        )

        data_dir = Path(self.hparams.data_dir)
        ia_df = pd.read_csv(
            data_dir / 'IA.tsv',
            sep='\t',
            header=None,
            index_col=None,
            names=['term', 'weight'],
        )
        self.term_to_weight = dict(zip(ia_df['term'], ia_df['weight']))

        train_non_propagated_terms_df = pd.read_csv(
            Path(self.hparams.train_terms_path),
            sep='\t',
        )

        # train_terms_df contains propagated terms
        train_terms_df = pd.read_csv(
            Path(self.hparams.train_propagated_terms_path),
            sep='\t'
        )
        
        self.ncbi_taxa = NCBITaxa(update=False)
        if self.hparams.val_dir is not None:
            logger.info('val_dir is provided, fold is ignored')
            val_terms_df = pd.read_csv(
                Path(self.hparams.val_dir) / 'validation_propagated_terms.tsv',
                sep='\t',
            )
            val_protein_ids = set(val_terms_df['EntryID'])
            train_protein_ids = set(train_terms_df['EntryID'])

            if self.hparams.use_cafa6_metrics:
                exclude_terms_df=train_terms_df.loc[
                    train_terms_df['EntryID'].isin(val_protein_ids)
                ].copy()
            else:
                exclude_terms_df=None

            self.cafa_metrics = CAFAMetrics(
                data_dir,
                val_terms_df,
                exclude_terms_df=exclude_terms_df,
                propagate_gt=False,
            )
            train_examples_df = pd.DataFrame(
                {'protein_id': sorted(train_protein_ids)}
            )
            val_examples_df = pd.DataFrame(
                {'protein_id': sorted(val_protein_ids)}
            )
        else:
            fold_df = pd.read_csv(self.hparams.fold_path)
            
            train_protein_ids = set(
                fold_df.loc[fold_df['fold'] != self.hparams.fold, 'protein_id']
            )
            val_protein_ids = set(
                fold_df.loc[fold_df['fold'] == self.hparams.fold, 'protein_id']
            )
            assert not train_protein_ids.intersection(val_protein_ids)

            val_terms_df = train_terms_df.loc[
                train_terms_df['EntryID'].isin(val_protein_ids)
            ].copy()
            self.cafa_metrics = CAFAMetrics(
                data_dir, val_terms_df, propagate_gt=False
            )

            train_examples_df = fold_df.loc[
                fold_df['protein_id'].isin(train_protein_ids)
            ].copy()
            val_examples_df = fold_df.loc[
                fold_df['protein_id'].isin(val_protein_ids)
            ].copy()
            del fold_df
        logger.info(f'num_train_examples={len(train_examples_df)}')
        logger.info(f'num_val_examples={len(val_examples_df)}')
        logger.info(
            'common train and val proteins='
            f'{len(train_protein_ids.intersection(val_protein_ids))}'
        )
        logger.info(
            f'val proteins only={len(val_protein_ids.difference(train_protein_ids))}'
        )

        logger.info(f'train terms={len(set(train_terms_df["term"]))}')
        logger.info(f'val terms={len(set(val_terms_df["term"]))}')
        logger.info(
            'common train and val terms='
            f'{len(set(train_terms_df["term"]).intersection(set(val_terms_df["term"])))}'
        )
        logger.info(
            f'val terms only={len(set(val_terms_df["term"]).difference(set(train_terms_df["term"])))}'
        )

        self.eval_metrics = EvalMetrics(
            data_dir,
            train_terms_df,
            val_terms_df,
        )

        filtered_terms = sorted(self.cafa_metrics.term_to_subontology.keys())
        if self.subontology is not None:
            filtered_terms = sorted(
                [
                    term
                    for term in filtered_terms
                    if self.cafa_metrics.term_to_subontology[term]
                    == self.subontology
                ]
            )

        filtered_so_term_pairs = [
            (self.cafa_metrics.term_to_subontology[term].name, term)
            for term in filtered_terms
        ]
        filtered_so_term_pairs = sorted(filtered_so_term_pairs)
        self.filtered_terms = [term for _, term in filtered_so_term_pairs]
        logger.info(f'num_filtered_terms={len(self.filtered_terms)}')

        bp_index, cc_index, mf_index = None, None, None
        for term_idx, (so, _) in enumerate(filtered_so_term_pairs):
            if so == Subontology.BP.name and bp_index is None:
                bp_index = term_idx
            if so == Subontology.CC.name and cc_index is None:
                cc_index = term_idx
            if so == Subontology.MF.name and mf_index is None:
                mf_index = term_idx
        self.subontology_ranges = (
            bp_index,
            cc_index,
            mf_index,
            len(self.filtered_terms),
        )
        logger.info(f'subontology_ranges={self.subontology_ranges}')

        subontology_edges_counts = {
            so.name: len(so_edges)
            for so, so_edges in self.cafa_metrics.subontology_edges.items()
        }
        logger.info(f'{subontology_edges_counts=}')
        filtered_terms_set = set(self.filtered_terms)
        filtered_subontology_edges = {
            so: [
                edge
                for edge in so_edges
                if edge[0] in filtered_terms_set
                and edge[1] in filtered_terms_set
            ]
            for so, so_edges in self.cafa_metrics.subontology_edges.items()
        }
        filtered_term_to_index = {
            term: term_index
            for term_index, term in enumerate(self.filtered_terms)
        }
        self.filtered_subontology_edges = {
            so: [
                (
                    filtered_term_to_index[edge[0]] - so_start_index,
                    filtered_term_to_index[edge[1]] - so_start_index,
                )
                for edge in filtered_subontology_edges[so]
            ]
            # The order is important subontology_ranges must be ordered
            # the same ways as Subontology
            for so, so_start_index in zip(
                Subontology, self.subontology_ranges[:-1]
            )
        }

        filtered_subontology_edges_counts = {
            so.name: len(so_edges)
            for so, so_edges in self.filtered_subontology_edges.items()
        }
        logger.info(f'{filtered_subontology_edges_counts=}')

        raw_filtered_terms_weights = np.array(
            [self.term_to_weight[term] for term in self.filtered_terms]
        )
        self.filtered_terms_weights = np.sqrt(raw_filtered_terms_weights)
        self.filtered_terms_weights[self.filtered_terms_weights < 0.5] = 0.5

        self.multi_label_binarizer = MultiLabelBinarizer(
            classes=self.filtered_terms, sparse_output=True
        )
        # fit should be called to construct multi_label_binarizer.classes_
        self.multi_label_binarizer.fit(None)

        if not is_predict:
            train_examples_df = self.fill_examples_target(
                train_examples_df, train_terms_df, train_non_propagated_terms_df
            )
            val_examples_df = self.fill_examples_target(
                val_examples_df, val_terms_df
            )

        combined_examples_df = pd.concat(
            [train_examples_df, val_examples_df], ignore_index=True
        )
        self.populate_examples_data(combined_examples_df)
        train_examples_df = combined_examples_df.iloc[
            : len(train_examples_df)
        ].copy()
        val_examples_df = combined_examples_df.iloc[
            len(train_examples_df) :
        ].copy()

        self.fit_preprocessors(train_examples_df)
        combined_examples_df = pd.concat(
            [train_examples_df, val_examples_df], ignore_index=True
        )
        self.preprocess_examples_data(combined_examples_df)
        train_examples_df = combined_examples_df.iloc[
            : len(train_examples_df)
        ].copy()
        val_examples_df = combined_examples_df.iloc[
            len(train_examples_df) :
        ].copy()
        del combined_examples_df

        if is_predict and self.hparams.predict_proteins_path is not None:
            val_examples_df = pd.read_csv(self.hparams.predict_proteins_path)
            self.populate_examples_data(val_examples_df)
            self.preprocess_examples_data(val_examples_df)
            val_examples_df = val_examples_df[self.dataset_columns]

        if not is_predict:
            self.dataset_columns += ['target', 'non_propagated_target']
        self.train_dataset = self.get_train_dataset(
            train_examples_df[self.dataset_columns].to_dict(orient='records')
        )
        logger.info(f'train_dataset size={len(self.train_dataset)}')

        self.val_dataset = self.get_val_dataset(
            val_examples_df[self.dataset_columns].to_dict(orient='records')
        )
        logger.info(f'val_dataset size={len(self.val_dataset)}')

    def get_train_dataset(self, examples: List[Dict[str, Any]]) -> Dataset:
        pass

    def get_val_dataset(self, examples: List[Dict[str, Any]]) -> Dataset:
        pass

    def fill_examples_target(
        self,
        examples_df: pd.DataFrame,
        terms_df: pd.DataFrame,
        non_propagated_terms_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        protein_terms_df = (
            terms_df.groupby(['EntryID'])['term'].apply(list).to_frame()
        )
        num_examples_before = len(examples_df)
        examples_df = examples_df.merge(
            protein_terms_df,
            how='inner',
            left_on='protein_id',
            right_on='EntryID',
        )
        assert len(examples_df) == num_examples_before, (
            f'Different number of examples after merging terms: '
            f' before={num_examples_before}, after={len(examples_df)}'
        )

        targets = self.multi_label_binarizer.transform(
            examples_df['term'].to_list()
        )
        # a sparse matrix with multi-label target for each example
        examples_df['target'] = [protein_target for protein_target in targets]

        if non_propagated_terms_df is not None:
            protein_non_propagated_terms_df = (
                non_propagated_terms_df.groupby(['EntryID'])['term']
                .apply(list)
                .to_frame()
            )
            examples_df = examples_df.merge(
                protein_non_propagated_terms_df,
                how='inner',
                left_on='protein_id',
                right_on='EntryID',
                suffixes=('', '_non_propagated'),
            )
            assert len(examples_df) == num_examples_before, (
                f'Different number of examples after merging non-propagated '
                f'terms: before={num_examples_before}, '
                f'after={len(examples_df)}'
            )
            non_propagated_targets = self.multi_label_binarizer.transform(
                examples_df['term_non_propagated'].to_list()
            )
            examples_df = examples_df.drop(
                columns=[
                    col
                    for col in examples_df.columns
                    if col.endswith('_non_propagated')
                ]
            )
            examples_df['non_propagated_target'] = [
                protein_target for protein_target in non_propagated_targets
            ]

        return examples_df

    def populate_examples_data(self, examples_df: pd.DataFrame) -> None:
        train_seq_by_protein_id = {}
        if self.hparams.train_seqs_path is not None:
            for seq_record in SeqIO.parse(
                self.hparams.train_seqs_path, 'fasta'
            ):
                train_protein_id = extract_protein_id_from_fasta(seq_record.id)
                train_seq_by_protein_id[train_protein_id] = seq_record

        test_seq_by_protein_id = {}
        if self.hparams.test_seqs_path is not None:
            for seq_record in SeqIO.parse(self.hparams.test_seqs_path, 'fasta'):
                test_protein_id = extract_protein_id_from_fasta(seq_record.id)
                if 'OX=' not in seq_record.description:
                    # The format of description is different for test sequences, unify
                    # it to have the same processing afterwards.
                    description_protein_id, taxon_id = (
                        seq_record.description.split(' ')
                    )
                    assert test_protein_id == description_protein_id
                    seq_record.description = f'{test_protein_id} OX={taxon_id}'
                test_seq_by_protein_id[test_protein_id] = seq_record

        examples_df['seq_record'] = examples_df['protein_id'].map(
            lambda protein_id: (
                train_seq_by_protein_id[protein_id]
                if protein_id in train_seq_by_protein_id
                else test_seq_by_protein_id[protein_id]
            )
        )
        examples_df['taxon_id'] = examples_df['seq_record'].map(
            lambda seq_rec: parse_taxon_id(seq_rec.description)
        )
        _, obsolete_taxon_id_map = self.ncbi_taxa._translate_merged(
            set(examples_df['taxon_id'])
        )
        # map obsolete taxon ids to their current ids
        examples_df['taxon_id'] = examples_df['taxon_id'].map(
            lambda taxon_id: obsolete_taxon_id_map.get(taxon_id, taxon_id)
        )

        unique_taxon_ids = sorted(set(examples_df['taxon_id']))
        logger.info(f'unique taxon ids in examples={len(unique_taxon_ids)}')
        unique_taxon_ids_str = ','.join(
            [f'"{taxon_id}"' for taxon_id in unique_taxon_ids]
        )
        # See self.ncbi_taxa.get_lineage(taxid)
        taxon_lineage_query = self.ncbi_taxa.db.execute(
            "SELECT taxid, track FROM species WHERE taxid IN (%s);"
            % unique_taxon_ids_str
        )
        lineage_by_taxon_id = {
            int(taxon_id): lineage_str
            for taxon_id, lineage_str in taxon_lineage_query.fetchall()
        }
        examples_df['taxon_lineage_ids'] = examples_df['taxon_id'].map(
            lambda taxon_id: lineage_by_taxon_id[taxon_id]
        )
        examples_df['norm_seq_len'] = examples_df['seq_record'].map(
            lambda seq_rec: len(seq_rec.seq)
        )
        examples_df['norm_seq_len'] = (
            np.clip(examples_df['norm_seq_len'], a_min=None, a_max=10000)
            / 10000
        )

    def fit_preprocessors(self, train_examples_df: pd.DataFrame):
        self.taxon_encoder = LabelEncoder()
        all_taxon_ids = {
            int(taxon_id)
            for taxon_lineage_ids in train_examples_df['taxon_lineage_ids']
            for taxon_id in taxon_lineage_ids.split(',')
        }
        all_taxon_ids.add(UNKNOWN_TAXON_ID)
        self.taxon_encoder.fit(sorted(all_taxon_ids))
        self.num_taxons = len(self.taxon_encoder.classes_)
        logger.info(f'num_taxons={self.num_taxons}')

    def preprocess_examples_data(self, examples_df: pd.DataFrame):
        known_taxon_ids = set(self.taxon_encoder.classes_)
        taxon_encoder_map = {
            label: label_idx
            for label_idx, label in enumerate(
                self.taxon_encoder.classes_.tolist()
            )
        }
        assert taxon_encoder_map[UNKNOWN_TAXON_ID] == 0
        examples_df['taxon_lineage_indices'] = [
            [
                taxon_encoder_map[
                    (
                        int(taxon_id)
                        if int(taxon_id) in known_taxon_ids
                        else UNKNOWN_TAXON_ID
                    )
                ]
                for taxon_id in taxon_lineage_ids.split(',')
            ]
            for taxon_lineage_ids in examples_df['taxon_lineage_ids']
        ]

    def get_collate_fn(self):
        return None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.get_collate_fn(),
            multiprocessing_context=(
                'fork' if torch.backends.mps.is_available() else None
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.get_collate_fn(),
            multiprocessing_context=(
                'fork' if torch.backends.mps.is_available() else None
            ),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.get_collate_fn(),
            multiprocessing_context=(
                'fork' if torch.backends.mps.is_available() else None
            ),
        )


def parse_taxon_id(description: str):
    match = re.search(r'OX=(\d+)', description)
    return int(match.group(1)) if match else UNKNOWN_TAXON_ID
