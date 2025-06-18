# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 10:01:53 2025

@author: YUEYU11
"""
from huggingface_hub import login
login(token="")

from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor
import torch

ds = load_dataset("mbazaNLP/kinyarwanda-tts-dataset") #transformer fairseq

ds = ds.cast_column("audio", Audio(sampling_rate=16000))

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def prepare(batch):
    audio = batch["audio"]

    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    with processor.as_target_processor():
        labels = processor(batch["sentence"]).input_ids

    batch["input_values"] = inputs.input_values[0]
    batch["labels"] = torch.tensor(labels)
    return batch

ds = ds.map(prepare, remove_columns=ds["train"].column_names)

from transformers import Wav2Vec2ForCTC, TrainingArguments, Trainer
import evaluate

wer_metric = evaluate.load("wer")

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base", ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id)

training_args = TrainingArguments(
    output_dir="./asr_ckpt",
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,
    learning_rate=3e-4,
    logging_dir="./logs",
    fp16=True # int4
)

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics
)

trainer.train()

import matplotlib.pyplot as plt
import seaborn as sns

model.eval()
with torch.no_grad():
    out = model(input_values=ds["test"][0]["input_values"].unsqueeze(0), output_hidden_states=True)
    features = out.hidden_states[-1].squeeze(0)  # shape: [T, D]

sns.heatmap(features.cpu().numpy().T, cmap="viridis")
plt.title("Last Layer Hidden Representation")
plt.xlabel("Time")
plt.ylabel("Feature dim")
plt.show()

#Connectionist Temporal Classification Loss
import numpy as np

def add_noise(batch, snr_db=10):
    audio = batch["audio"]["array"]
    rms = np.sqrt(np.mean(audio ** 2))
    noise = np.random.normal(0, rms / (10**(snr_db / 20)), audio.shape)
    batch["audio"]["array"] = audio + noise
    return batch

noisy_ds = load_dataset("mbazaNLP/kinyarwanda-tts-dataset", split="test")
noisy_ds = noisy_ds.map(add_noise)

