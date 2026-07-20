from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import T5Config, WhisperConfig, T5Model, WhisperModel
from transformers.modeling_outputs import Seq2SeqModelOutput

from .spectrogram import MelSpectrogram
from ..tokenizer import Tokenizer

LABEL_IGNORE_ID = -100


@dataclass
class OsuClassifierOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    feature_vector: Optional[torch.FloatTensor] = None


def get_backbone_model(args, tokenizer: Tokenizer):
    if args.model.name.startswith("google/t5"):
        config = T5Config.from_pretrained(args.model.name)
    elif args.model.name.startswith("openai/whisper"):
        config = WhisperConfig.from_pretrained(args.model.name)
    else:
        raise NotImplementedError

    config.vocab_size = tokenizer.vocab_size

    if hasattr(args.model, "overwrite"):
        for k, v in args.model.overwrite.items():
            assert hasattr(config, k), f"config does not have attribute {k}"
            setattr(config, k, v)

    if hasattr(args.model, "add_config"):
        for k, v in args.model.add_config.items():
            assert not hasattr(config, k), f"config already has attribute {k}"
            setattr(config, k, v)

    if args.model.name.startswith("google/t5"):
        model = T5Model(config)
    elif args.model.name.startswith("openai/whisper"):
        config.use_cache = False
        config.num_mel_bins = config.d_model
        config.pad_token_id = tokenizer.pad_id
        config.max_source_positions = args.data.src_seq_len // 2
        config.max_target_positions = args.data.tgt_seq_len
        model = WhisperModel(config)
    else:
        raise NotImplementedError

    return model, config.d_model


class OsuClassifier(nn.Module):
    __slots__ = [
        "spectrogram",
        "decoder_embedder",
        "encoder_embedder",
        "transformer",
        "style_embedder",
        "num_classes",
        "input_features",
        "projector",
        "classifier",
        "vocab_size",
        "loss_fn",
    ]

    def __init__(self, args: DictConfig, tokenizer: Tokenizer):
        super().__init__()

        self.transformer, d_model = get_backbone_model(args, tokenizer)
        self.num_classes = tokenizer.num_classes
        self.input_features = args.model.input_features

        self.decoder_embedder = nn.Embedding(tokenizer.vocab_size, d_model)
        self.decoder_embedder.weight.data.normal_(mean=0.0, std=1.0)

        self.spectrogram = MelSpectrogram(
            args.model.spectrogram.sample_rate, args.model.spectrogram.n_fft,
            args.model.spectrogram.n_mels, args.model.spectrogram.hop_length
        )

        self.encoder_embedder = nn.Linear(args.model.spectrogram.n_mels, d_model)

        self.projector = nn.Linear(d_model, args.model.classifier_proj_size)
        self.classifier = nn.Linear(args.model.classifier_proj_size, tokenizer.num_classes)

        self.vocab_size = tokenizer.vocab_size
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
            self,
            frames: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            **kwargs
    ) -> OsuClassifierOutput:
        """
        frames: B x L_encoder x mel_bins, float32
        decoder_input_ids: B x L_decoder, int64
        beatmap_id: B, int64
        encoder_outputs: B x L_encoder x D, float32
        """

        frames = self.spectrogram(frames)  # (N, L, M)
        inputs_embeds = self.encoder_embedder(frames)
        decoder_inputs_embeds = self.decoder_embedder(decoder_input_ids)

        if self.input_features:
            input_features = torch.swapaxes(inputs_embeds, 1, 2) if inputs_embeds is not None else None
            # noinspection PyTypeChecker
            base_output: Seq2SeqModelOutput = self.transformer.forward(input_features=input_features,
                                                                       decoder_inputs_embeds=decoder_inputs_embeds,
                                                                       **kwargs)
        else:
            base_output = self.transformer.forward(inputs_embeds=inputs_embeds,
                                                   decoder_inputs_embeds=decoder_inputs_embeds,
                                                   **kwargs)

        # Get logits
        hidden_states = self.projector(base_output.last_hidden_state)
        pooled_output = hidden_states.mean(dim=1)

        logits = self.classifier(pooled_output)
        loss = None

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_classes), labels.view(-1))

        return OsuClassifierOutput(
            loss=loss,
            logits=logits,
            encoder_last_hidden_state=base_output.encoder_last_hidden_state,
            decoder_last_hidden_state=base_output.last_hidden_state,
            feature_vector=pooled_output
        )
