import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from jsonargparse import CLI
from tqdm import tqdm

from cafa.core.constants import SUBMISSION_MIN_PROBABILITY
from cafa.core.metrics import CAFAMetrics, Subontology
from cafa.core.common.logging import config_logging
from cafa.core.common.utils import load_pickle

logger = logging.getLogger(__name__)


def ensemble_predictions(
    data_dir: str, prediction_paths: list[str], output_dir: str
) -> None:
    assert not os.path.exists(
        output_dir
    ), f'Output dir {output_dir} already exists.'
    os.makedirs(output_dir, exist_ok=False)
    config_logging(output_dir)

    data_dir = Path(data_dir)
    # Needed for CAFAMetrics initialization
    dummy_gt_terms_df = pd.read_csv(
        data_dir / 'Train' / 'train_terms.tsv', sep='\t'
    )
    cafa_metrics = CAFAMetrics(data_dir, dummy_gt_terms_df, propagate_gt=False)

    prediction_path = prediction_paths[0]
    log_config_info(prediction_path)
    is_compressed = prediction_path.endswith('.gz')
    epoch_outputs = load_pickle(prediction_path, compress=is_compressed)
    ensemble_protein_ids = epoch_outputs['protein_id']
    ensemble_prediction_per_subontology = epoch_outputs['prediction']

    for prediction_path in tqdm(prediction_paths[1:]):
        log_config_info(prediction_path)
        is_compressed = prediction_path.endswith('.gz')
        epoch_outputs = load_pickle(prediction_path, compress=is_compressed)
        assert (
            ensemble_protein_ids == epoch_outputs['protein_id']
        ), 'Protein IDs do not match between predictions.'
        for subontology in Subontology:
            ensemble_prediction_per_subontology[subontology] += epoch_outputs[
                'prediction'
            ][subontology]

    for subontology in Subontology:
        ensemble_prediction_per_subontology[subontology] /= len(
            prediction_paths
        )
    ensemble_epoch_outputs = {
        'protein_id': ensemble_protein_ids,
        'prediction': ensemble_prediction_per_subontology,
    }
    ensemble_sub_df = epoch_outputs_to_submission_df(
        cafa_metrics,
        ensemble_epoch_outputs,
    )
    ensemble_sub_df['prob'] = ensemble_sub_df['prob'].clip(upper=0.999)
    ensemble_sub_df['prob'] = (
        ensemble_sub_df['prob'].round(3).astype(np.float32)
    )

    prob_thresh = SUBMISSION_MIN_PROBABILITY
    logger.info(
        f'Before prob filtering {len(ensemble_sub_df)=} with prob > {prob_thresh}'
    )
    ensemble_sub_df = ensemble_sub_df.loc[
        (ensemble_sub_df['prob'] > prob_thresh), :
    ]
    ensemble_sub_df = ensemble_sub_df.sort_values(
        by=['protein_id', 'prob', 'term'], ascending=[True, False, True]
    )
    logger.info(
        f'After prob filtering {len(ensemble_sub_df)=} with prob > {prob_thresh}'
    )

    ensemble_sub_df.to_csv(
        os.path.join(output_dir, 'submission.tsv'),
        sep='\t',
        index=False,
        header=False,
    )


def epoch_outputs_to_submission_df(
    cafa_metrics: CAFAMetrics, epoch_outputs: dict[str, Any]
) -> pd.DataFrame:
    ensemble_sub_df = cafa_metrics.to_protein_term_prediction_probs(
        epoch_outputs['prediction'], epoch_outputs['protein_id']
    )
    return ensemble_sub_df


def log_config_info(prediction_path):
    config_path = Path(prediction_path).parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"seed_everything={config['seed_everything']}")
        logger.info(f"ckpt_path={config['ckpt_path']}")
        logger.info(
            f"stochastic_predict={config['data']['init_args']['stochastic_predict']}"
        )


if __name__ == "__main__":
    CLI(ensemble_predictions, as_positional=False)
