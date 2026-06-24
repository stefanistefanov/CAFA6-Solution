import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import safetensors
import scipy.sparse
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.trainer.states import TrainerFn
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW

from cafa.core.constants import SUBMISSION_MIN_PROBABILITY
from cafa.core.eval_metrics import KnowledgeCategory
from cafa.core.metrics import Subontology, get_max_metrics
from cafa.core.common.utils import save_pickle

logger = logging.getLogger(__name__)


class CAFAModule(LightningModule):
    def __init__(
        self,
        dropout: float = 0.0,
        lr: float = 1e-3,
        min_lr: float = 1e-5,
        weight_decay: float = 1e-7,
        save_embeddings: bool = False,
        save_submission: bool = True,
        save_validation_epoch_outputs: bool = False,
        model_checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.loss = nn.BCEWithLogitsLoss(reduction='none')

        self.validation_step_outputs = []
        self.predict_step_outputs = []

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if self.hparams.save_embeddings and stage != TrainerFn.PREDICTING:
            raise ValueError(
                'Saving of embeddings is only supported for predict'
            )
        self.num_labels = len(self.trainer.datamodule.filtered_terms)
        self.subontology_ranges = self.trainer.datamodule.subontology_ranges
        self.stochastic_predict = (
            self.trainer.datamodule.hparams.stochastic_predict
        )
        logger.info(f'{self.stochastic_predict=}')
        self.use_cafa6_metrics = self.trainer.datamodule.hparams.use_cafa6_metrics
        logger.info(f'{self.use_cafa6_metrics=}')

        self.register_buffer(
            'label_weights',
            torch.tensor(
                self.trainer.datamodule.filtered_terms_weights,
                dtype=self.dtype,
                device=self.device,
            ),
        )

        self.register_buffer(
            'bp_edges',
            torch.tensor(
                self.trainer.datamodule.filtered_subontology_edges[
                    Subontology.BP
                ],
                dtype=torch.long,
                device=self.device,
            ).t(),
        )

        self.register_buffer(
            'cc_edges',
            torch.tensor(
                self.trainer.datamodule.filtered_subontology_edges[
                    Subontology.CC
                ],
                dtype=torch.long,
                device=self.device,
            ).t(),
        )

        self.register_buffer(
            'mf_edges',
            torch.tensor(
                self.trainer.datamodule.filtered_subontology_edges[
                    Subontology.MF
                ],
                dtype=torch.long,
                device=self.device,
            ).t(),
        )

        self.cafa_metrics = self.trainer.datamodule.cafa_metrics
        self.eval_metrics = self.trainer.datamodule.eval_metrics

    def on_fit_start(self):
        logger.info(f'model={self}')
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info(
            f'Total parameters: {total_params}, '
            f'Trainable parameters: {trainable_params}'
        )

    def on_train_epoch_start(self) -> None:
        self.train_epoch_start_time = time.time()

    def on_train_epoch_end(
        self,
    ) -> None:
        train_epoch_time = time.time() - self.train_epoch_start_time
        self.log('train_epoch_time', train_epoch_time)

    def on_validation_epoch_start(self) -> None:
        self.valid_epoch_start_time = time.time()

    def on_validation_epoch_end(self) -> None:
        epoch_outputs = self.process_epoch_outputs(self.validation_step_outputs)
        # Free memory
        self.validation_step_outputs.clear()

        if self.hparams.save_validation_epoch_outputs:
            save_pickle(
                epoch_outputs,
                os.path.join(
                    self.trainer.default_root_dir,
                    f'validation_epoch_outputs.pkl.gz',
                ),
                compress=True,
            )
            if self.hparams.save_submission:
                self.save_submission(epoch_outputs)

        if self.use_cafa6_metrics:
            eval_metrics_dict = self.eval_metrics.compute(
                epoch_outputs['prediction'], epoch_outputs['protein_id']
            )
        else:
            eval_metrics_dict = {}
        cafa5_metrics_df, cafa5_max_metric_dfs = self.cafa_metrics.compute(
            epoch_outputs['prediction'], epoch_outputs['protein_id']
        )
        eval_metrics_dict['cafa5'] = (cafa5_metrics_df, cafa5_max_metric_dfs)

        max_metrics_key = 'wf'
        cafa5_scores = []
        cafa6_scores = []
        cafa6_prefixes = []
        for prefix, (metrics_df, max_metric_dfs) in eval_metrics_dict.items():
            max_metrics = get_max_metrics(max_metric_dfs, max_metrics_key)
            is_cafa6_prefix = prefix in {
                KnowledgeCategory.NO.value,
                KnowledgeCategory.LIMITED.value,
                KnowledgeCategory.PARTIAL.value,
            }
            if is_cafa6_prefix:
                cafa6_prefixes.append(prefix)
            for metric_name, metric_value in max_metrics.items():
                self.log(
                    f'val_{prefix}/{metric_name}',
                    float(metric_value),
                    prog_bar=False,
                )
                metric_name_match = re.fullmatch(
                    rf'max_{max_metrics_key}/.+/{max_metrics_key}', metric_name
                )
                if prefix == 'cafa5' and metric_name_match:
                    cafa5_scores.append(metric_value)
                elif prefix in cafa6_prefixes and metric_name_match:
                    cafa6_scores.append(metric_value)
        
        assert len(cafa5_scores) == 3
        cafa5_score = np.mean(cafa5_scores)
        self.log('val_cafa5_score', float(cafa5_score), prog_bar=False)
        if self.use_cafa6_metrics:
            cafa6_scores_log = '/'.join(cafa6_prefixes) + ' scores: '
            cafa6_scores_log += ', '.join(f'{score:.4f}' for score in cafa6_scores)
            logger.info(f'epoch={self.current_epoch} {cafa6_scores_log}')
            assert len(cafa6_scores) == 9
            cafa6_score = np.mean(cafa6_scores)
            val_score_epoch = cafa6_score
        else:
            val_score_epoch = cafa5_score
        self.log('val_score_epoch', val_score_epoch, prog_bar=True)
        logger.info(
            f'epoch={self.current_epoch}'
            f' val_score_epoch={val_score_epoch:.4f}'
        )

        valid_epoch_time = time.time() - self.valid_epoch_start_time
        self.log('val_epoch_time', valid_epoch_time)

    def on_predict_epoch_end(self) -> None:
        logger.info('on_predict_epoch_end ...')
        logger.info('Processing epoch_outputs ...')
        epoch_outputs = self.process_epoch_outputs(self.predict_step_outputs)
        logger.info('Processing epoch_outputs done.')
        # Free memory
        self.predict_step_outputs.clear()

        if self.hparams.save_embeddings:
            self.save_embeddings(epoch_outputs)
        else:
            logger.info('Saving predict_epoch_outputs ...')
            save_pickle(
                epoch_outputs,
                os.path.join(
                    self.trainer.default_root_dir,
                    f'predict_epoch_outputs.pkl',
                ),
                compress=False,
            )
            logger.info('Saving predict_epoch_outputs done.')
            if self.hparams.save_submission:
                logger.info('Saving submission ...')
                self.save_submission(epoch_outputs)
                logger.info('Saving submission done.')
        logger.info('on_predict_epoch_end done.')

    def save_embeddings(self, epoch_outputs):
        embedding_by_protein_id = {}
        for protein_id, protein_embedding in zip(
            epoch_outputs['protein_id'], epoch_outputs['embedding']
        ):
            embedding_by_protein_id[protein_id] = protein_embedding
        safetensors.torch.save_file(
            embedding_by_protein_id,
            os.path.join(
                self.trainer.default_root_dir,
                'embeddings.safetensors',
            ),
        )

    def save_submission(self, epoch_outputs) -> None:
        prediction_probs_per_subontology: Dict[Subontology, csr_matrix] = (
            epoch_outputs['prediction']
        )
        prediction_protein_ids: List[str] = epoch_outputs['protein_id']
        sub_df = self.cafa_metrics.to_protein_term_prediction_probs(
            prediction_probs_per_subontology, prediction_protein_ids
        )
        prob_thresh = SUBMISSION_MIN_PROBABILITY
        logger.info(
            f'Before prob filtering {len(sub_df)=} with prob > {prob_thresh}'
        )
        sub_df = sub_df.loc[(sub_df['prob'] > prob_thresh), :]
        logger.info(
            f'After prob filtering {len(sub_df)=} with prob > {prob_thresh}'
        )
        sub_df.to_csv(
            os.path.join(self.trainer.default_root_dir, 'submission.tsv'),
            sep='\t',
            index=False,
            header=False,
        )

    def process_epoch_outputs(self, batch_outputs):
        epoch_outputs = {
            'protein_id': [],
        }
        if self.hparams.save_embeddings:
            epoch_outputs['embedding'] = []
        else:
            epoch_outputs['prediction'] = {
                subontology: None for subontology in Subontology
            }
            if 'target' in batch_outputs[0]:
                epoch_outputs['target'] = []

        batch_preds_per_subontology = {
            subontology: [] for subontology in Subontology
        }
        for batch_output in batch_outputs:
            for output_key in epoch_outputs:
                if output_key == 'prediction':
                    for subontology in Subontology:
                        batch_preds_per_subontology[subontology].append(
                            batch_output[output_key][subontology]
                        )
                else:
                    epoch_outputs[output_key].extend(batch_output[output_key])
        for subontology in Subontology:
            if len(batch_preds_per_subontology[subontology]) > 0:
                epoch_outputs['prediction'][subontology] = scipy.sparse.vstack(
                    batch_preds_per_subontology[subontology]
                )
        return epoch_outputs

    def step_common(self, batch, prefix, log_loss_on_step):
        targets = batch['target']
        batch_size = len(targets)
        batch_outputs = self(batch)
        logits = batch_outputs['logits']
        loss = []
        for start_idx, end_idx in zip(
            self.subontology_ranges[:-1],
            self.subontology_ranges[1:],
        ):
            so_logits = logits[:, start_idx:end_idx]
            so_targets = targets[:, start_idx:end_idx]
            no_gt_mask = torch.any(so_targets, dim=1).to(so_targets.dtype)
            so_loss = self.loss(so_logits, so_targets)
            so_label_weights = self.label_weights[start_idx:end_idx]
            so_loss = so_loss * so_label_weights
            so_loss = so_loss * no_gt_mask.unsqueeze(1)
            so_loss = so_loss.mean()

            loss.append(so_loss)
        loss = torch.mean(torch.stack(loss))

        self.log(
            f'{prefix}_loss',
            loss,
            on_step=log_loss_on_step,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        prediction_probs = torch.sigmoid(logits)
        targets = targets.to(torch.int32)
        step_outputs = {
            'loss': loss,
            'prediction_probs': prediction_probs,
            'targets': targets,
        }

        return step_outputs


    def training_step(self, batch, batch_idx):
        step_outputs = self.step_common(
            batch, prefix='train', log_loss_on_step=True
        )
        return {'loss': step_outputs['loss']}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        step_outputs = self.step_common(
            batch, prefix=f'val', log_loss_on_step=False
        )

        prediction_probs = (
            step_outputs['prediction_probs']
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        targets = step_outputs['targets'].detach().cpu().numpy()

        prediction = self.cafa_metrics.to_prediction_probs_per_subontology(
            prediction_probs,
            prediction_terms=self.trainer.datamodule.filtered_terms,
            propagate=True,
        )

        val_step_output = {
            'prediction': prediction,
            'target': targets,
            'protein_id': batch['protein_id'],
        }

        self.validation_step_outputs.append(val_step_output)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        batch_outputs = self(batch)
        prediction_probs = torch.sigmoid(batch_outputs['logits'])
        prediction_probs = (
            prediction_probs.detach().cpu().numpy().astype(np.float32)
        )

        pred_step_output = {
            'protein_id': batch['protein_id'],
        }
        if self.hparams.save_embeddings:
            embeddings = batch_outputs['embeddings'].detach().cpu()
            pred_step_output['embedding'] = embeddings
        else:
            prediction = self.cafa_metrics.to_prediction_probs_per_subontology(
                prediction_probs,
                prediction_terms=self.trainer.datamodule.filtered_terms,
                propagate=True,
            )
            pred_step_output['prediction'] = prediction

        self.predict_step_outputs.append(pred_step_output)

        if self.hparams.save_embeddings:
            num_predicted_proteins = sum(
                len(step_output['protein_id'])
                for step_output in self.predict_step_outputs
            )
            batch_size = len(pred_step_output['protein_id'])
            interim_save_step = (10000 // batch_size) * batch_size
            if num_predicted_proteins % interim_save_step == 0:
                interim_epoch_outputs = self.process_epoch_outputs(
                    self.predict_step_outputs
                )
                # No clearing of self.predict_step_outputs as this is an
                # intermediate saving
                self.save_embeddings(interim_epoch_outputs)


    def calculate_metrics(
        self,
        epoch_outputs: Dict[str, List],
        prefix: Optional[str] = None,
    ):
        targets = np.stack(epoch_outputs['target'], axis=0)
        predictions = np.stack(epoch_outputs['prediction'], axis=0)
        predictions = predictions > 0.5
        f1_micro = f1_score(targets, predictions, average='micro')
        metrics = {'f1_micro': f1_micro}
        if prefix is not None:
            metrics = {
                f'{prefix}_{metric_name}': metric_value
                for metric_name, metric_value in metrics.items()
            }
        return metrics

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        num_training_steps = total_training_steps(self.trainer)
        logger.info(f'num_training_steps={num_training_steps}')
       
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1 * self.hparams.lr,
            pct_start=0.05,
            div_factor=30,
            final_div_factor=1e4,
            total_steps=num_training_steps,
            anneal_strategy='cos',
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }

def total_training_steps(trainer: Trainer) -> int:
    if trainer.max_steps != -1:
        return trainer.max_steps

    limit_batches = trainer.limit_train_batches
    batches = len(trainer.datamodule.train_dataloader())
    batches = (
        min(batches, limit_batches)
        if isinstance(limit_batches, int)
        else int(limit_batches * batches)
    )
    effective_accum = trainer.accumulate_grad_batches * trainer.num_devices
    return math.ceil(batches / effective_accum) * trainer.max_epochs
