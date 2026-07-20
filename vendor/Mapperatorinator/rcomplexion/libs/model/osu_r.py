from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from ..tokenizer import Tokenizer


class OsuR(nn.Module):
    def __init__(self, args: DictConfig, tokenizer: Tokenizer):
        super().__init__()
        self.vocab_size = tokenizer.vocab_size_in
        self.num_labels = tokenizer.vocab_size_out
        self.sequence_length = args.data.src_seq_len
        self.hidden_size = args.model.hidden_size

        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

        # Simple sequence classification model
        self.model = nn.Sequential(
            nn.Embedding(self.vocab_size, self.hidden_size),
            nn.Flatten(),
            nn.Linear(self.hidden_size * self.sequence_length, self.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_labels)
        )

    def forward(
            self,
            input_ids: torch.LongTensor,
            labels: Optional[torch.LongTensor] = None,
            sample_weights: Optional[torch.FloatTensor] = None,
            **kwargs
    ) -> SequenceClassifierOutput:
        """
        Args:
            input_ids: (N, L) tensor of input token ids
            labels: (N) tensor of target token ids
            sample_weights: (N) tensor of sample weights
        """

        # Embed the input tokens
        logits = self.model(input_ids)

        # Calculate loss
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if sample_weights is not None:
                loss *= sample_weights
            loss = loss.mean()

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

