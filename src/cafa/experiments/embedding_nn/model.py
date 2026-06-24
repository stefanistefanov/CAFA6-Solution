import logging

import torch
import torch.nn.functional as F
from torch import nn

from cafa.core.model import CAFAModule

logger = logging.getLogger(__name__)

class TermEvidenceEncoder(nn.Module):
    """
    A 2-layer Conv1D network to project term-evidence one-hot encoded matrix
    to term logits.
    """

    def __init__(
        self,
        num_terms: int,
        num_evidences: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.num_terms = num_terms
        self.num_evidences = num_evidences
        self.term_evidence_proj1 = nn.Conv1d(
            self.num_evidences, hidden_dim, kernel_size=1
        )
        self.term_evidence_dropout = nn.Dropout(p=dropout)
        self.term_evidence_proj2 = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, term_indices, evidence_indices) -> torch.Tensor:
        batch_size = term_indices.shape[0]
        device = term_indices.device

        # One-hot matrix term_evidence_matrix: (batch_size, num_evidences, num_terms)
        x = torch.zeros(
            batch_size,
            self.num_evidences,
            self.num_terms,
            device=device,
            dtype=self.term_evidence_proj1.weight.dtype,
        )

        valid_mask = (evidence_indices != -1) & (term_indices != -1)
        batch_indices = (
            torch.arange(x.shape[0], device=device)
            .unsqueeze(1)
            .expand_as(evidence_indices)[valid_mask]
        )
        valid_evidence_indices = evidence_indices[valid_mask]
        valid_term_indices = term_indices[valid_mask]
        x[batch_indices, valid_evidence_indices, valid_term_indices] = 1

        # Projection: (batch_size, num_evidences, num_terms) -> (batch_size, hidden_dim, num_terms)
        x = F.gelu(self.term_evidence_proj1(x))
        x = self.term_evidence_dropout(x)
        # Projection: (batch_size, hidden_dim, num_terms) -> (batch_size, 1, num_terms)
        x = self.term_evidence_proj2(x)
        x = x.squeeze(1)  # (batch_size, num_terms)
        # zero index corresponds to the UNKNOWN_INPUT_TERM_IDX, shift it,
        # so that it is aligned with the terms output logits
        return x[:, 1:]


class TermEvidenceEmbeddingEncoder(nn.Module):
    def __init__(
        self, num_terms, num_evidences, embedding_dim, nhead=8, dropout=0.1
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.term_embedding = nn.Embedding(
            num_terms + 1, embedding_dim, padding_idx=0
        )
        self.evidence_embedding = nn.Embedding(
            num_evidences + 1, embedding_dim, padding_idx=0
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=1024,
            norm_first=True,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1
        )

    def forward(self, term_indices, evidence_indices):
        """
        Args:
            term_indices: (batch_size, padded_length) with padding value -1
            evidence_indices: (batch_size, padded_length) with padding value -1

        Returns:
            pooled_output: (batch_size, embedding_dim)
        """

        # Shift indices by 1 so that -1 becomes 0 (padding_idx)
        term_indices_shifted = term_indices + 1
        evidence_indices_shifted = evidence_indices + 1

        # embeddings: (batch_size, padded_length, embedding_dim)
        term_emb = self.term_embedding(term_indices_shifted)
        evidence_emb = self.evidence_embedding(evidence_indices_shifted)

        combined_emb = term_emb + evidence_emb

        # (batch_size, padded_length)
        padding_mask = term_indices == -1

        # (batch_size, padded_length, embedding_dim)
        transformer_output = self.transformer_encoder(
            combined_emb, src_key_padding_mask=padding_mask
        )

        # Mean pooling: ignore padding positions
        valid_mask = (~padding_mask).unsqueeze(-1).float()

        # Sum over padded_length dimension and divide by number of valid positions
        # (batch_size, embedding_dim)
        masked_output = transformer_output * valid_mask
        sum_output = masked_output.sum(dim=1)
        count_valid = valid_mask.sum(dim=1).clamp(min=1)
        pooled_output = sum_output / count_valid

        return pooled_output


class TaxonEmbeddingEncoder(nn.Module):
    def __init__(self, num_taxons, embedding_dim, nhead=8, dropout=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.taxon_embedding = nn.Embedding(
            num_taxons + 1, embedding_dim, padding_idx=0
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=1024,
            norm_first=True,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1
        )

    def forward(self, taxon_indices):
        # Shift indices by 1 so that -1 becomes 0 (padding_idx)
        taxon_indices_shifted = taxon_indices + 1

        # embeddings: (batch_size, padded_length, embedding_dim)
        taxon_emb = self.taxon_embedding(taxon_indices_shifted)

        # (batch_size, padded_length)
        padding_mask = taxon_indices == -1

        # (batch_size, padded_length, embedding_dim)
        transformer_output = self.transformer_encoder(
            taxon_emb, src_key_padding_mask=padding_mask
        )

        # Mean pooling: ignore padding positions
        valid_mask = (~padding_mask).unsqueeze(-1).float()

        # Sum over padded_length dimension and divide by number of valid positions
        # (batch_size, embedding_dim)
        masked_output = transformer_output * valid_mask
        sum_output = masked_output.sum(dim=1)
        count_valid = valid_mask.sum(dim=1).clamp(min=1)
        pooled_output = sum_output / count_valid

        return pooled_output


class LogitsGate(nn.Module):
    def __init__(self, features_dim: int, dropout: float):
        super().__init__()
        self.features_proj = nn.Linear(features_dim, 126)
        self.conv1 = nn.Conv1d(126 + 2, 256, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(256, 1, kernel_size=1)

    def forward(self, x, y, features):
        features = F.gelu(self.features_proj(features))
        features = features.unsqueeze(2).expand(
            -1, -1, x.shape[1]
        )
        g = torch.cat(
            [x.unsqueeze(1), y.unsqueeze(1), features], dim=1
        )
        g = F.gelu(self.conv1(g))
        g = self.dropout(g)
        g = self.conv2(g)
        g = torch.sigmoid(g)
        # (batch_size, num_terms)
        g = g.squeeze(1)  
        return x + g * y


class Classifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_features=in_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_dim, out_features=out_dim),
        )

        self.logits_gate = LogitsGate(features_dim=in_dim, dropout=0.5)

    def forward(self, features, classifier_term_evidence_logits):
        logits = self.classifier(features)
        logits = self.logits_gate(
            logits, classifier_term_evidence_logits, features
        )
        return logits


class EmbeddingNN(CAFAModule):
    def setup(self, stage: str) -> None:
        super().setup(stage)
        norm_seq_len_dim = 1
        taxon_embedding_dim = 512
        term_evidence_embedding_dim = 512

        self.term_evidence_encoder = TermEvidenceEncoder(
            num_terms=self.trainer.datamodule.input_terms_size,
            num_evidences=self.trainer.datamodule.input_evidences_size,
            hidden_dim=128,
            dropout=0.5,
        )
        self.term_evidence_embedding_encoder = TermEvidenceEmbeddingEncoder(
            num_terms=self.trainer.datamodule.input_terms_size,
            num_evidences=self.trainer.datamodule.input_evidences_size,
            embedding_dim=term_evidence_embedding_dim,
            nhead=8,
            dropout=0.1,
        )

        self.taxon_embedding_encoder = TaxonEmbeddingEncoder(
            num_taxons=self.trainer.datamodule.num_taxons,
            embedding_dim=taxon_embedding_dim,
            nhead=8,
            dropout=0.1,
        )

        self.layer_in_dims = self.trainer.datamodule.embedding_dims
        layer_out_dims = len(self.layer_in_dims) * [1024]
        self.layers = nn.ModuleList()
        for layer_in_dim, layer_out_dim in zip(
            self.layer_in_dims, layer_out_dims, strict=True
        ):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=layer_in_dim,
                        out_features=layer_out_dim,
                    ),
                    nn.GELU(),
                )
            )

        classifier_in_dim = (
            norm_seq_len_dim
            + sum(layer_out_dims)
            + taxon_embedding_dim
            + term_evidence_embedding_dim
        )
        self.classifiers = nn.ModuleList()
        classifiers_hidden_dims = (3072, 1024, 1536)
        for classifier_idx, (start_idx, end_idx) in enumerate(
            zip(
                self.subontology_ranges[:-1],
                self.subontology_ranges[1:],
                strict=True,
            )
        ):
            clf_hidden_dim = classifiers_hidden_dims[classifier_idx]
            self.classifiers.append(
                Classifier(
                    in_dim=classifier_in_dim,
                    hidden_dim=clf_hidden_dim,
                    out_dim=end_idx - start_idx,
                )
            )

        self.load_model_state()

    def load_model_state(self):
        if self.hparams.model_checkpoint_path is None:
            return
        checkpoint = torch.load(
            self.hparams.model_checkpoint_path,
            weights_only=False,
        )
        model_load_result = self.load_state_dict(
            checkpoint['state_dict'], strict=False
        )
        logger.info(f'{model_load_result=}')

    def forward(self, batch):
        evidence_indices = batch['evidence_indices']
        original_term_indices = batch['original_term_indices']
        term_evidence_logits = self.term_evidence_encoder(
            term_indices=original_term_indices,
            evidence_indices=evidence_indices,
        )
        term_indices = batch['term_indices']
        term_evidence_embedding = self.term_evidence_embedding_encoder(
            term_indices=term_indices,
            evidence_indices=evidence_indices,
        )

        taxon_embedding = self.taxon_embedding_encoder(
            batch['taxon_lineage_indices']
        )

        x = [batch['norm_seq_len']]
        start_layer_in_dim = 0
        for layer_idx, layer in enumerate(self.layers):
            layer_input = batch['embedding'][
                :,
                start_layer_in_dim : start_layer_in_dim
                + self.layer_in_dims[layer_idx],
            ]
            x.append(layer(layer_input))
            start_layer_in_dim += self.layer_in_dims[layer_idx]
        x.append(taxon_embedding)
        x.append(term_evidence_embedding)
        x = torch.cat(x, dim=1)
        del term_evidence_embedding, taxon_embedding

        logits = []
        for classifier, start_idx, end_idx in zip(
            self.classifiers,
            self.subontology_ranges[:-1],
            self.subontology_ranges[1:],
            strict=True,
        ):
            clf_logits = classifier(
                x, term_evidence_logits[:, start_idx:end_idx]
            )
            logits.append(clf_logits)
        logits = torch.concat(logits, dim=1)
        return {'logits': logits}
