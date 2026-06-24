import logging
from enum import Enum
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix

from cafa.core.metrics import CAFAMetrics, Subontology

logger = logging.getLogger(__name__)


class KnowledgeCategory(Enum):
    NO = 'no_knowledge'
    LIMITED = 'limited_knowledge'
    PARTIAL = 'partial_knowledge'


def select_outputs_by_protein_ids(
    prediction_probs_per_subontology: dict[Subontology, csr_matrix],
    prediction_protein_ids: list[str],
    allowed_protein_ids: list[str],
) -> tuple[dict[Subontology, csr_matrix], list[str]]:
    protein_id_to_index_ordered = {
        protein_id: idx
        for idx, protein_id in enumerate(prediction_protein_ids)
        if protein_id in allowed_protein_ids
    }
    selected_prediction = {}
    protein_indices = list(protein_id_to_index_ordered.values())
    for subontology in Subontology:
        selected_prediction[subontology] = prediction_probs_per_subontology[
            subontology
        ][protein_indices, :]
    selected_prediction_protein_ids = list(protein_id_to_index_ordered.keys())
    return selected_prediction, selected_prediction_protein_ids


class EvalMetrics:
    def __init__(
        self,
        data_dir: Path,
        train_terms_df: pd.DataFrame,
        val_terms_df: pd.DataFrame,
    ):
        logger.info(
            f'val_terms_df={len(val_terms_df)}'
            f' val_proteins={val_terms_df["EntryID"].nunique()}'
            f' val_terms={val_terms_df["term"].nunique()}'
        )
        self.no_knowledge_protein_ids = sorted(
            set(val_terms_df['EntryID'].unique())
            - set(train_terms_df['EntryID'].unique())
        )
        no_knowledge_val_terms_df = (
            val_terms_df.loc[
                val_terms_df['EntryID'].isin(self.no_knowledge_protein_ids)
            ]
            .copy()
            .reset_index(drop=True)
        )
        logger.info(
            f'no_knowledge_val_terms_df={len(no_knowledge_val_terms_df)}'
            f' no_knowledge_proteins={len(self.no_knowledge_protein_ids)}'
            f' no_knowledge_terms={no_knowledge_val_terms_df["term"].nunique()}'
        )

        knowledge_val_terms_df = val_terms_df.loc[
            ~val_terms_df['EntryID'].isin(self.no_knowledge_protein_ids)
        ].copy()

        limited_knowledge_val_terms_df = (
            knowledge_val_terms_df.merge(
                train_terms_df.drop(columns=['term']),
                on=['EntryID', 'aspect'],
                how='left',
                indicator=True,
            )
            .query('_merge == "left_only"')
            .drop('_merge', axis=1)
            .reset_index(drop=True)
        )
        self.limited_knowledge_protein_ids = sorted(
            limited_knowledge_val_terms_df['EntryID'].unique()
        )
        logger.info(
            f'limited_knowledge_val_terms_df={len(limited_knowledge_val_terms_df)}'
            f' limited_knowledge_proteins={len(self.limited_knowledge_protein_ids)}'
            f' limited_knowledge_terms={limited_knowledge_val_terms_df["term"].nunique()}'
        )

        remaining_knowledge_val_terms_df = (
            knowledge_val_terms_df.merge(
                limited_knowledge_val_terms_df, how='left', indicator=True
            )
            .query('_merge == "left_only"')
            .drop('_merge', axis=1)
            .reset_index(drop=True)
        )

        new_val_terms_not_in_train_df = (
            remaining_knowledge_val_terms_df.merge(
                train_terms_df, how='left', indicator=True
            )
            .query('_merge == "left_only"')
            .drop('_merge', axis=1)
            .reset_index(drop=True)
        )

        partial_knowledge_val_terms_df = remaining_knowledge_val_terms_df.loc[
            remaining_knowledge_val_terms_df['EntryID'].isin(
                new_val_terms_not_in_train_df['EntryID']
            )
        ].copy()
        self.partial_knowledge_protein_ids = sorted(
            partial_knowledge_val_terms_df['EntryID'].unique()
        )
        logger.info(
            f'partial_knowledge_val_terms_df={len(partial_knowledge_val_terms_df)}'
            f' partial_knowledge_proteins={len(self.partial_knowledge_protein_ids)}'
            f' partial_knowledge_terms={partial_knowledge_val_terms_df["term"].nunique()}'
        )
        
        assert limited_knowledge_val_terms_df.merge(
            train_terms_df, on=['EntryID', 'aspect', 'term'], how='inner'
        ).empty, (
            'There are overlapping terms between limited_knowledge_val_terms_df '
            'and train_terms_df.'
        )

        self.cafa_metrics_no_knowledge = CAFAMetrics(
            data_dir,
            no_knowledge_val_terms_df.copy(),
            exclude_terms_df=None,
            propagate_gt=False,
        )

        self.cafa_metrics_limited_knowledge = CAFAMetrics(
            data_dir,
            limited_knowledge_val_terms_df.copy(),
            exclude_terms_df=None,
            propagate_gt=False,
        )

        self.cafa_metrics_partial_knowledge = CAFAMetrics(
            data_dir,
            partial_knowledge_val_terms_df.copy(),
            exclude_terms_df=train_terms_df.loc[
                train_terms_df['EntryID'].isin(
                    self.partial_knowledge_protein_ids
                )
            ].copy(),
            propagate_gt=False,
        )

    def compute(
        self,
        prediction_probs_per_subontology: dict[Subontology, csr_matrix],
        prediction_protein_ids: list[str],
    ):
        assert set(prediction_protein_ids).issubset(
            set(
                self.no_knowledge_protein_ids
                + self.limited_knowledge_protein_ids
                + self.partial_knowledge_protein_ids
            )
        )
        no_knowledge_pred, no_knowledge_pred_protein_ids = (
            select_outputs_by_protein_ids(
                prediction_probs_per_subontology,
                prediction_protein_ids,
                self.no_knowledge_protein_ids,
            )
        )
        no_knowledge_metrics_df, no_knowledge_max_metric_dfs = (
            self.cafa_metrics_no_knowledge.compute(
                no_knowledge_pred,
                no_knowledge_pred_protein_ids,
            )
        )

        limited_knowledge_pred, limited_knowledge_pred_protein_ids = (
            select_outputs_by_protein_ids(
                prediction_probs_per_subontology,
                prediction_protein_ids,
                self.limited_knowledge_protein_ids,
            )
        )
        limited_knowledge_metrics_df, limited_knowledge_max_metric_dfs = (
            self.cafa_metrics_limited_knowledge.compute(
                limited_knowledge_pred,
                limited_knowledge_pred_protein_ids,
            )
        )
        partial_knowledge_pred, partial_knowledge_pred_protein_ids = (
            select_outputs_by_protein_ids(
                prediction_probs_per_subontology,
                prediction_protein_ids,
                self.partial_knowledge_protein_ids,
            )
        )
        partial_knowledge_metrics_df, partial_knowledge_max_metric_dfs = (
            self.cafa_metrics_partial_knowledge.compute(
                partial_knowledge_pred,
                partial_knowledge_pred_protein_ids,
            )
        )
        return {
            KnowledgeCategory.NO.value: (
                no_knowledge_metrics_df,
                no_knowledge_max_metric_dfs,
            ),
            KnowledgeCategory.LIMITED.value: (
                limited_knowledge_metrics_df,
                limited_knowledge_max_metric_dfs,
            ),
            KnowledgeCategory.PARTIAL.value: (
                partial_knowledge_metrics_df,
                partial_knowledge_max_metric_dfs,
            ),
        }
