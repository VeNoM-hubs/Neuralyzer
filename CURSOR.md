# CURSOR.md

# Project

Faithful reimplementation of the paper

"A Multimodal Framework for Dementia Detection via Linguistic and Acoustic Representation Learning"

Goal:
Reproduce the architecture, training pipeline, and ablation studies as closely as possible to the paper.

Primary reference:
A Multimodal Framework for Dementia Detection via Linguistic and Acoustic Representation Learning (2026)

DO NOT simplify or replace architectural components unless explicitly approved.

---

# Development Philosophy

This project aims for reproducibility.

Whenever the paper specifies something, implement it exactly.

Whenever the paper does NOT specify something, DO NOT assume.

Instead:

1. Stop.
2. Explain what is missing.
3. Ask me.
4. Wait for confirmation.

Never silently replace unspecified components with your preferred implementation.

---

# Architecture (Paper)

Input

- Speech (.wav)
- Transcript

Speech Branch

Speech
↓

Split into fixed 10-second chunks

↓

Pretrained HuBERT

↓

Take last hidden layer
+
second-last hidden layer

↓

Element-wise sum

↓

Attentive Statistics Pooling

↓

Weighted Mean

+

Weighted Standard Deviation

↓

Concatenate

↓

Chunk embedding

↓

Average all chunk embeddings

↓

Linear projection

↓

Speech embedding

Text Branch

Transcript

↓

BERT Tokenizer

↓

Pretrained BERT

↓

CLS token

↓

Text embedding

Multimodal

Speech embedding

+

Text embedding

↓

Audio-Text Fusion

↓

Fused embedding

↓

Classifier

↓

2-class logits

Loss

CrossEntropy

+

λ × MINE

---

# Project Structure

src/

    datasets/
    models/
    modules/
    losses/
    trainer/
    evaluation/
    utils/
    configs/

train.py

inference.py

requirements.txt

README.md

---

# Modules

datasets/

Responsible for

- loading audio
- loading transcript
- labels
- train/validation split
- chunk creation

models/

Contains

HubERT encoder

BERT encoder

Multimodal model

modules/

Attentive Statistics Pooling

AT-Fusion

MINE

Classifier

losses/

CrossEntropy

MINE loss

trainer/

training loop

checkpointing

early stopping

scheduler

evaluation/

metrics

confusion matrix

ablation scripts

---

# Paper Components

Must implement

✓ HuBERT

✓ BERT

✓ Attentive Statistics Pooling

✓ Audio-Text Fusion

✓ MINE

✓ CrossEntropy

✓ Final classifier

---

# Required Ablation Studies

Pooling

- Mean
- Max
- ASP

Speech Model

- HuBERT
- wav2vec2
- XLS-R

Lambda

0

0.1

0.2

0.25

0.3

Fusion

Concatenation

GMU

MUTAN

MFB

MFH

BLOCK

AT-Fusion

HuBERT Layers

Last layer

Last two

Last three

---

# Training Settings (Specified)

Batch size

8

Early stopping

8 epochs

Scheduler

StepLR

step_size = 4

gamma = 0.1

Loss

CrossEntropy

+

0.25 × MINE

Framework

PyTorch

---

# Things NOT Specified By The Paper

The following implementation details are NOT given.

Cursor MUST ask before implementing.

## Optimizer

Paper does not specify.

Examples

Adam

AdamW

SGD

Ask first.

---

## Learning Rate

Paper does not specify.

Ask before coding.

---

## Weight Decay

Not specified.

Ask.

---

## Betas

Not specified.

Ask.

---

## Gradient Clipping

Not specified.

Ask.

---

## Random Seed

Not specified.

Ask.

---

## HuBERT Checkpoint

Paper only says

Pretrained HuBERT

Not specified

Examples

facebook/hubert-base-ls960

hubert-large

etc.

Ask first.

---

## BERT Checkpoint

Paper only says

Pretrained BERT

Not specified

Examples

bert-base-uncased

bert-large

Ask first.

---

## Audio Sampling Rate

Not specified.

16 kHz?

8 kHz?

Ask first.

---

## Audio Resampling

Not specified.

Ask.

---

## Audio Normalization

Not specified.

Ask.

---

## Silence Removal

Not specified.

Ask.

---

## Padding Strategy

Not specified.

Ask.

---

## Maximum Audio Length

Not specified.

Ask.

---

## BERT Maximum Sequence Length

Not specified.

Ask.

---

## Tokenizer Options

Not specified.

Ask.

---

## Truncation Strategy

Not specified.

Ask.

---

## Dropout Probability

Paper shows Dropout in the architecture but does not specify probability.

Ask.

---

## Linear Layer Sizes

Paper figure shows a projection/classifier but does not specify every hidden dimension except the final 2-class output.

Ask before assuming dimensions that are not explicitly stated.

---

## MINE Network

Paper only says

Simple linear layers
+
ReLU

Does NOT specify

- number of layers
- hidden size
- output dimension

Ask before implementing.

---

## AT-Fusion

Paper provides equations.

Implementation details not specified.

Ask before coding.

---

## Initialization

Paper does not specify.

Ask.

---

## Mixed Precision

Not specified.

Ask.

---

## Gradient Accumulation

Not specified.

Ask.

---

## Checkpoint Saving

Not specified.

Ask.

---

## Validation Split

Paper says 65/35.

Need implementation details.

Ask.

---

## Logging

Not specified.

TensorBoard?

Weights & Biases?

CSV?

Ask.

---

## Data Augmentation

Paper mentions none.

Do NOT add augmentation unless explicitly requested.

---

# Coding Rules

Keep modules independent.

Never hardcode paths.

Use dataclasses or YAML configs.

Use type hints.

Document tensor shapes.

Every forward() must contain tensor shape comments.

Example

Audio

[B,T]

↓

HuBERT

[B,L,768]

↓

ASP

[B,1536]

↓

Projection

[B,768]

---

# Before Every Major Module

Cursor should first explain

1. Why the module exists.

2. What tensor enters.

3. What tensor exits.

4. Which equation from the paper it implements.

Then write the code.

---

# If Anything Is Missing

Do NOT invent.

Ask me first.

Paper fidelity is higher priority than convenience.

---

# ========================================
# DEVELOPMENT DATASET POLICY
# ========================================

The final target dataset for this project is ADReSS (and later PROCESS-2), exactly as used in the paper.

However, during development, ADReSS may not yet be available.

Therefore:

- The architecture MUST remain exactly as described in the paper.
- No architectural component may be removed or simplified because a temporary dataset is being used.
- Only the dataset implementation is allowed to change.

The development dataset is ONLY for validating:

- data loading
- preprocessing
- HuBERT pipeline
- BERT pipeline
- Attentive Statistics Pooling
- MINE
- AT-Fusion
- classifier
- training loop
- evaluation pipeline

The dataset MUST expose the following interface:

```python
Sample = {
    "audio": waveform,
    "transcript": string,
    "label": integer
}
```

The model code must never depend on a specific dataset.

Changing from a temporary dataset to ADReSS should only require replacing the Dataset class.

---

# ========================================
# WHEN USING A TEMPORARY DATASET
# ========================================

If a temporary dataset differs from ADReSS, Cursor must:

1. Inform me exactly what differs.

Examples:

- different labels
- different folder structure
- different transcript format
- different sampling rate
- missing transcripts
- different train/validation split

2. Explain whether the difference affects

- architecture
- preprocessing
- training
- evaluation

3. Ask for approval before adapting the dataset loader.

Only the dataset loader may change.

The model architecture must remain unchanged.

---

# ========================================
# NO ARCHITECTURAL SHORTCUTS
# ========================================

Do NOT replace or remove any component because the temporary dataset is different.

For example, do NOT:

- remove MINE
- remove AT-Fusion
- remove ASP
- replace HuBERT
- replace BERT
- skip transcript processing
- replace multimodal learning with audio-only
- replace multimodal learning with text-only

Instead, explain why the temporary dataset cannot support that component and ask me how to proceed.

---

# ========================================
# DEVELOPMENT GOAL
# ========================================

The purpose of the temporary dataset is ONLY to verify that the implementation is correct.

The repository should be written so that once ADReSS is available, switching datasets requires changing only the Dataset implementation, not the model, trainer, or loss functions.

---

# ========================================
# PAPER REPRODUCTION RULE
# ========================================

Whenever implementing a section of the paper:

1. First identify the exact figure, equation, or paragraph being implemented.

2. Explain your interpretation.

3. If there are multiple valid interpretations, STOP and ask me which one to use.

4. Do not infer implementation details unless the paper clearly implies them.

5. Every assumption must be explicitly stated before writing code.

Never silently choose one interpretation when the paper is ambiguous.
