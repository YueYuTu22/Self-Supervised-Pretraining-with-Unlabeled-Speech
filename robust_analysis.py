# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 20:13:28 2025

@author: YUEYU11
"""

import os
import pandas as pd
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import matplotlib.pyplot as plt
import jiwer
import librosa
from sklearn.model_selection import train_test_split

# ========== PATH CONFIGURATION ==========
base_dir = r"C:/Yue/ASR/dataset/kinyarwanda-tts-dataset"
csv_path = os.path.join(base_dir, "tts-dataset.csv")
audio_dir = os.path.join(base_dir, "audio")
proc_dir = r"C:/Yue/ASR/project7_1_kinya-20250720T171957Z-1-001/project7_1_kinya/processor/wav2vec2-base-kinya-final"
model_dir = r"C:/Yue/ASR/project7_1_kinya-20250720T171957Z-1-001/project7_1_kinya/wav2vec2-base-kinya-final"

# ========== 1. DATA LOADING AND PREPROCESSING ==========
# Read CSV and construct DataFrame with full audio path and corresponding text
data = []
with open(csv_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        idx = line.find(' ')
        if idx == -1:
            continue
        file_id = line[:idx].replace('"', '')
        text = line[idx:].strip().strip('"')
        if file_id.startswith('TTS_'):
            wav_name = 'TTS ' + file_id[4:] + '.wav'
        else:
            wav_name = file_id + '.wav'
        wav_path = os.path.join(audio_dir, wav_name)
        data.append({'audio_path': wav_path.replace("\\", "/"), 'text': text})

df = pd.DataFrame(data)

# Split dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
test_df = test_df.reset_index(drop=True)
print(f"[DEBUG] test_df shape: {test_df.shape}")

# ========== 2. LOAD MODEL AND PROCESSOR ==========
# Load trained Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained(proc_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_dir)
model.eval()
if torch.cuda.is_available():
    model.to("cuda")
    print("[DEBUG] Using GPU")
else:
    print("[DEBUG] Using CPU")

# ========== 3. LOAD ALL AUDIO WAVEFORMS ==========
# Use librosa to load audio at 16 kHz
def load_waveform(audio_path):
    try:
        arr, sr = librosa.load(audio_path, sr=16000)
        return arr
    except Exception as e:
        print(f"[ERROR] Failed to read: {audio_path}, {e}")
        return np.zeros(16000, dtype=np.float32)

test_df['speech'] = test_df['audio_path'].apply(load_waveform)

# ========== 4. ROBUSTNESS EVALUATION FUNCTION ==========
def add_noise(wave, snr_db):
    """
    Add Gaussian noise to a waveform at a specified SNR (dB).
    :param wave: np.array, the input waveform.
    :param snr_db: float, target SNR in decibels.
    :return: np.array, noisy waveform.
    """
    rms_signal = np.sqrt(np.mean(wave ** 2))
    snr = 10 ** (snr_db / 10)
    rms_noise = np.sqrt(rms_signal ** 2 / snr)
    noise = np.random.normal(0, rms_noise, wave.shape)
    return wave + noise

# Define SNR values to test, from clean (high SNR) to very noisy (low SNR)
snr_list = [30, 20, 10, 5, 0, -5]  # in dB
wers = []
cers = []
sample_size = 100  # Number of test samples for each SNR setting

# Evaluate model robustness for each SNR level
for snr_db in snr_list:
    preds = []
    refs = []
    print(f"[INFO] Processing SNR={snr_db} dB ...")
    # Fix random_state to avoid error for negative values
    sample_df = test_df.sample(sample_size, random_state=snr_db + 100 if snr_db < 0 else snr_db)
    for i, row in sample_df.iterrows():
        # Add noise to audio
        noisy = add_noise(row['speech'], snr_db)
        # Feature extraction
        input_values = processor(noisy, sampling_rate=16000).input_values
        input_tensor = torch.tensor(input_values)
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)
        if torch.cuda.is_available():
            input_tensor = input_tensor.to("cuda")
        # Forward pass (no gradients needed)
        with torch.no_grad():
            logits = model(input_tensor).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_text = processor.batch_decode(pred_ids.cpu().numpy())[0]
        preds.append(pred_text)
        refs.append(row['text'])
    # Compute WER/CER for this SNR
    wer = jiwer.wer(refs, preds)
    cer = jiwer.cer(refs, preds)
    print(f"SNR={snr_db} dB, WER={wer:.2f}, CER={cer:.2f}")
    wers.append(wer)
    cers.append(cer)

# ========== 5. PLOT WER/CER vs SNR ==========
plt.figure(figsize=(8, 5))
plt.plot(snr_list, wers, marker='o', label="WER")
plt.plot(snr_list, cers, marker='s', label="CER")
plt.xlabel("SNR (dB)")
plt.ylabel("Error Rate")
plt.title("Model Robustness vs Noise (WER/CER vs SNR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


