import logging
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import default_collate
from transformers import AutoTokenizer, DataCollatorWithPadding, T5Tokenizer

from cafa.core.datamodule import ProteinDataModule
from cafa.experiments.sequence_embedder.dataset import ProteinSequenceDataset

logger = logging.getLogger(__name__)


class ProteinSequenceDataModule(ProteinDataModule):
    def __init__(
        self,
        data_dir: str,
        train_terms_path: str,
        train_propagated_terms_path: str,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        fold_path: str,
        fold: int,
        train_seqs_path: str | None,
        test_seqs_path: str | None,
        model_name: str,
        max_sequence_len: int = 1024,
        predict_proteins_path: str = None,
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
        self.dataset_columns += ['sequence']

        if self.hparams.model_name.startswith('facebook/esm2'):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.model_name
            )
        elif self.hparams.model_name.startswith('Rostlab/ProstT5_fp16'):
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.model_name, do_lower_case=False
            )
        else:
            raise ValueError(f'Unknown model_name={self.hparams.model_name}')
        
        self.padding_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, pad_to_multiple_of=16, padding='longest'
        )

    def get_train_dataset(self, examples: List[Dict[str, Any]]):
        return ProteinSequenceDataset(
            examples,
            self.tokenizer,
            self.hparams.max_sequence_len,
            train=True,
            model_name=self.hparams.model_name,
        )

    def get_val_dataset(self, examples: List[Dict[str, Any]]):
        return ProteinSequenceDataset(
            examples,
            self.tokenizer,
            self.hparams.max_sequence_len,
            train=False,
            model_name=self.hparams.model_name,
        )

    def populate_examples_data(self, examples_df: pd.DataFrame):
        super().populate_examples_data(examples_df)
        examples_df['sequence'] = examples_df['seq_record'].map(
            lambda seq_rec: str(seq_rec.seq)
        )

    def get_collate_fn(self):
        return self.collate

    def collate(self, batch: List[Dict[str, Any]]):
        batch_to_pad = []
        for elem in batch:
            batch_to_pad.append({'input_ids': elem.pop('input_ids')})
        collated_batch = default_collate(batch)
        collated_padded_batch = self.padding_collator(batch_to_pad)
        collated_batch.update(**collated_padded_batch)
        collated_batch['input_ids'] = collated_batch['input_ids'].to(
            dtype=torch.int32
        )
        collated_batch['attention_mask'] = collated_batch['attention_mask'].to(
            dtype=torch.int16
        )
        if 'target' in collated_batch:
            collated_batch['target'] = collated_batch['target'].to(
                dtype=torch.float16
            )
        
        return collated_batch
