import logging
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from cafa.core.constants import UNKNOWN_INPUT_TERM_IDX, UNKNOWN_TAXON_ID

logger = logging.getLogger(__name__)


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        train: bool,
        stochastic_predict: bool,
        embedding_dims: List[int],
        subontology_ranges: tuple[int, ...],
    ):
        super().__init__()
        self.examples = examples
        self.train = train
        self.stochastic_predict = stochastic_predict
        self.embedding_dims = embedding_dims
        self.subontology_ranges = subontology_ranges

    def __getitem__(self, index):
        example = self.examples[index]
        embedding = example['embedding'].detach().clone()
        term_indices = np.array(example['term_indices'])
        original_term_indices = np.array(example['original_term_indices'])
        taxon_lineage_indices = example['taxon_lineage_indices']

        if self.train or self.stochastic_predict:
            if len(term_indices) > 0 and np.random.uniform() < 0.2:
                term_idxs_to_unknown = np.random.choice(
                    len(term_indices),
                    size=max(1, int(0.1 * len(term_indices))),
                    replace=False,
                )
                term_indices[term_idxs_to_unknown] = UNKNOWN_INPUT_TERM_IDX
                original_term_indices[term_idxs_to_unknown] = (
                    UNKNOWN_INPUT_TERM_IDX
                )

            # The unseen are more specific taxons, which are the first taxons
            # in the lineage. Keep last taxons and replace first taxons with 
            # UNKNOWN_TAXON_ID to simulate unseen taxons.
            if np.random.uniform() < 0.2:
                keep_last_n_taxons = np.random.randint(0, 6)
                taxon_lineage_indices = [UNKNOWN_TAXON_ID] * len(
                    taxon_lineage_indices[:keep_last_n_taxons]
                ) + taxon_lineage_indices[keep_last_n_taxons:]

        dtype = torch.float32
        item = {
            'embedding': embedding.to(dtype),
            'taxon_lineage_indices': torch.tensor(
                taxon_lineage_indices, dtype=torch.long
            ),
            'norm_seq_len': torch.tensor(
                [example['norm_seq_len']], dtype=dtype
            ),
            'protein_id': str(example['protein_id']),
            'original_term_indices': torch.from_numpy(original_term_indices).to(
                dtype=torch.long
            ),
            'term_indices': torch.from_numpy(term_indices).to(dtype=torch.long),
            'evidence_indices': torch.tensor(
                example['evidence_indices'], dtype=torch.long
            ),
            'num_input_terms': len(example['term_indices']),
        }
        if (
            self.train or self.stochastic_predict
        ) and np.random.uniform() < 0.9:
            self.dropout_features(item)

        if 'target' in example:
            target = example['target'].toarray()[0]
            item['target'] = torch.from_numpy(target).to(dtype)

        return item

    def dropout_features(self, item):
        features = [
            f'embedding_{i}' for i in range(len(self.embedding_dims))
        ] + [
            'taxon_lineage_indices',
            'original_term_indices',
            'term_indices',
        ]
        dropout_features = np.random.choice(features, size=2, replace=False)
        for dropout_feature in dropout_features:
            if dropout_feature.startswith('embedding_'):
                embedding_idx = int(dropout_feature[len('embedding_') :])
                start_idx = sum(self.embedding_dims[:embedding_idx])
                end_idx = start_idx + self.embedding_dims[embedding_idx]
                item['embedding'][start_idx:end_idx] = 0.0
            elif dropout_feature == 'taxon_lineage_indices':
                item[dropout_feature] = torch.tensor(
                    [UNKNOWN_TAXON_ID], dtype=torch.long
                )
            elif dropout_feature in {'original_term_indices', 'term_indices'}:
                item[dropout_feature] = torch.tensor(
                    [UNKNOWN_INPUT_TERM_IDX], dtype=torch.long
                )
            else:
                raise ValueError(f"Unknown dropout feature {dropout_feature}")

    def __len__(self):
        return len(self.examples)
