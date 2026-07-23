"""Batch collation: 10-second chunking + tokenization.

Realizes the paper's "divide each recording into fixed 10-second segments"
(Sec IV-b) plus BERT tokenization. Kept separate from both dataset and model:
the dataset yields raw ``Sample`` dicts, the model consumes tensors, and
chunking/tokenization live here (shared by all datasets).

Batching strategy for variable-length recordings:
  Each recording produces a variable number C_i of 10-second chunks. All chunks
  across the batch are flattened into one [sum(C_i), chunk_len] tensor so the
  speech encoder runs once; ``chunk_to_sample`` maps each chunk back to its
  recording so per-chunk embeddings can be averaged (paper: average chunk
  embeddings across all segments).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class MultimodalCollator:
    feature_extractor: object
    tokenizer: object
    sample_rate: int = 16000
    chunk_seconds: float = 10.0
    max_text_length: int = 512
    max_chunks_per_recording: Optional[int] = None

    @property
    def chunk_len(self) -> int:
        return int(round(self.sample_rate * self.chunk_seconds))

    def _chunk(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Split a 1-D waveform [T] into non-overlapping chunks of chunk_len.

        The final partial chunk is kept shorter; the feature extractor pads it
        and the attention mask excludes padded frames from pooling. Always
        returns at least one chunk. If ``max_chunks_per_recording`` is set, only
        the first N chunks are kept (bounds GPU memory for long recordings).
        """
        T = waveform.shape[0]
        clen = self.chunk_len
        if T == 0:
            return [torch.zeros(clen, dtype=torch.float32)]
        chunks = [waveform[i : i + clen] for i in range(0, T, clen)]
        if self.max_chunks_per_recording is not None:
            chunks = chunks[: self.max_chunks_per_recording]
        return chunks

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        all_chunks: List[torch.Tensor] = []
        chunk_to_sample: List[int] = []
        transcripts: List[str] = []
        labels: List[int] = []

        for sample_idx, sample in enumerate(batch):
            transcripts.append(sample["transcript"])
            labels.append(sample["label"])
            for chunk in self._chunk(sample["audio"]):
                all_chunks.append(chunk)
                chunk_to_sample.append(sample_idx)

        # --- Audio: normalize + pad chunks to equal length -------------------
        audio_inputs = self.feature_extractor(
            [c.numpy() for c in all_chunks],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values = audio_inputs["input_values"]  # [num_chunks, chunk_len]
        audio_attention_mask = audio_inputs.get(
            "attention_mask", torch.ones_like(input_values, dtype=torch.long)
        )  # [num_chunks, chunk_len]

        # --- Text: tokenize (pad to longest, truncate at max length) ---------
        text_inputs = self.tokenizer(
            transcripts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        return {
            "input_values": input_values,  # [num_chunks, chunk_len]
            "audio_attention_mask": audio_attention_mask,  # [num_chunks, chunk_len]
            "chunk_to_sample": torch.tensor(chunk_to_sample, dtype=torch.long),  # [num_chunks]
            "input_ids": text_inputs["input_ids"],  # [B, S]
            "text_attention_mask": text_inputs["attention_mask"],  # [B, S]
            "token_type_ids": text_inputs.get(
                "token_type_ids", torch.zeros_like(text_inputs["input_ids"])
            ),  # [B, S]
            "labels": torch.tensor(labels, dtype=torch.long),  # [B]
            "batch_size": len(batch),
        }
