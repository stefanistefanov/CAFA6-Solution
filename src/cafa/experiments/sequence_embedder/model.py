import logging

import torch
from transformers import AutoModel, T5EncoderModel

from cafa.core.model import CAFAModule


logger = logging.getLogger(__name__)


class ProteinSequenceEmbedder(CAFAModule):
    def setup(self, stage: str) -> None:
        super().setup(stage)
        model_name = self.trainer.datamodule.hparams.model_name
        if model_name.startswith('facebook/esm2'):
            self.base_model = AutoModel.from_pretrained(
                self.trainer.datamodule.hparams.model_name
            )
        elif model_name.startswith('Rostlab/ProstT5_fp16'):
            self.base_model = T5EncoderModel.from_pretrained(
                self.trainer.datamodule.hparams.model_name,
                dtype=torch.float16,
            )
        else:
            raise ValueError(f'Unknown model_name={model_name}')
        logger.info(f'config={self.base_model.config}')
        for param in self.base_model.parameters():
            param.requires_grad = False

        
    def forward(self, batch):
        attention_mask = batch['attention_mask']
        outputs = self.base_model(
            input_ids=batch['input_ids'],
            attention_mask=attention_mask,
        )
        embeddings = self.get_embeddings(outputs, attention_mask)
        model_outputs = {'logits': None}
        # dummy logits
        model_outputs['logits'] = torch.zeros(
            (embeddings.shape[0], self.num_labels),
            device=self.device,
            dtype=self.dtype,
        )
        model_outputs['embeddings'] = embeddings
       
        return model_outputs

    def get_embeddings(self, base_model_outputs, attention_mask):
        last_hidden_state = base_model_outputs[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
