import torch

from ..tokenizer import Event, EventType, Tokenizer


def create_sequences(tokens: torch.Tensor, src_seq_len: int, tokenizer: Tokenizer) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Create sequences from the tokenized event sequence.

    Args:
        tokens: The tokenized event sequence.
        src_seq_len: The source sequence length.
        tokenizer: The tokenizer to use.

    Returns:
        The sequences and labels.
    """
    sequences = []
    labels = []
    timed_events = [tokenizer.encode(Event(EventType.CIRCLE)), tokenizer.encode(Event(EventType.SLIDER_HEAD))]
    for i in range(src_seq_len + 1, len(tokens)):
        if tokens[i] not in timed_events:
            continue

        sequences.append(tokens[i - 1 - src_seq_len:i - 1])
        labels.append(tokens[i - 1])

    return sequences, labels


def tokenize_events(events: list[Event], tokenizer: Tokenizer) -> torch.Tensor:
    """Tokenize the event sequence.

    Args:
        events: The input events.
        tokenizer: The tokenizer to use.

    Returns:
        The tokenized events.
    """
    tokens = torch.empty(len(events), dtype=torch.long)
    for i, event in enumerate(events):
        tokens[i] = tokenizer.encode(event)
    return tokens
