"""
Custom BERT/RoBERTa token classifier for span identification.

DEPRECATED: This module is superseded by AutoModelForTokenClassification
loaded in hf_trainer.py. All active experiment scripts use the HuggingFace
Trainer path and never instantiate TokenClassifierForSpans. Retained for
reference only.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.span_identification.tokenization import LabelScheme, get_label2id, labels_to_spans


class TokenClassifierForSpans(nn.Module):
    """Token classification model for span identification."""

    def __init__(
        self,
        model_name: str,
        label_scheme: LabelScheme = "BIO",
        num_labels: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.label_scheme = label_scheme
        label2id = get_label2id(label_scheme)
        self.num_labels = num_labels or len(label2id)
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}

        config = AutoConfig.from_pretrained(model_name, num_labels=self.num_labels)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, self.num_labels)
        self.model_name = model_name

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        result: dict[str, Any] = {"logits": logits}
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
            loss = loss_fct(active_logits, active_labels)
            result["loss"] = loss
        return result

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "TokenClassifierForSpans":
        """Load from checkpoint path."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            meta = ckpt.get("meta", {})
            model = cls(
                model_name=meta.get("model_name", "bert-base-uncased"),
                label_scheme=meta.get("label_scheme", "BIO"),
                num_labels=meta.get("num_labels"),
                dropout=meta.get("dropout", 0.1),
            )
            model.load_state_dict(ckpt["model_state"])
            return model
        return ckpt

    def save_pretrained(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.state_dict(),
                "meta": {
                    "model_name": self.model_name,
                    "label_scheme": self.label_scheme,
                    "num_labels": self.num_labels,
                },
            },
            path,
        )


def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)
