
import os
import pandas as pd
import numpy as np
import librosa
from transformers import Wav2Vec2Processor
from sklearn.model_selection import train_test_split

# 1. Read and build DataFrame
base_dir = r"C:/Yue/ASR/dataset/kinyarwanda-tts-dataset"
csv_path = os.path.join(base_dir, "tts-dataset.csv")
audio_dir = os.path.join(base_dir, "audio")

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
print("[DEBUG] First 5 rows of DataFrame:")
print(df.head())

# 2. Split into train/test
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
print(f"[DEBUG] Train size: {len(train_df)}, Test size: {len(test_df)}")

# 3. Load processor
proc_dir = r"C:\Yue\ASR\project7_1_kinya-20250720T171957Z-1-001\project7_1_kinya\processor\wav2vec2-base-kinya-final"
processor = Wav2Vec2Processor.from_pretrained(proc_dir)

# 4. Manually load all audio waveforms
def load_waveform(audio_path):
    try:
        arr, sr = librosa.load(audio_path, sr=16000)
        return arr.tolist()
    except Exception as e:
        print(f"[ERROR] Failed to read: {audio_path}, {e}")
        return [0.0]*16000

train_df['speech'] = train_df['audio_path'].apply(load_waveform)
test_df['speech'] = test_df['audio_path'].apply(load_waveform)

# 5. Feature extraction (tokenization, etc.) is performed within DataFrame
def extract_input_values(waveform):
    return processor(waveform, sampling_rate=16000).input_values

def extract_labels(text):
    with processor.as_target_processor():
        return processor(text).input_ids

train_df['input_values'] = train_df['speech'].apply(extract_input_values)
train_df['labels'] = train_df['text'].apply(extract_labels)
test_df['input_values'] = test_df['speech'].apply(extract_input_values)
test_df['labels'] = test_df['text'].apply(extract_labels)

# 6. All further analysis, distribution statistics, and model training can be conducted directly with DataFrame
print(train_df.head())
print(test_df.head())

import matplotlib.pyplot as plt
from collections import Counter

# 1. Audio duration distribution (in seconds)
def plot_audio_duration(df, title="Audio Duration Distribution"):
    durations = [len(x) / 16000 for x in df['speech']]
    plt.figure(figsize=(8, 5))
    plt.hist(durations, bins=30, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of Samples")
    plt.grid(alpha=0.3)
    plt.show()
    print(f"[INFO] Mean duration: {np.mean(durations):.2f}s, Min: {np.min(durations):.2f}s, Max: {np.max(durations):.2f}s")

print("\n# === Audio Duration Distribution (train) ===")
plot_audio_duration(train_df, "Audio Duration Distribution (train)")

print("\n# === Audio Duration Distribution (test) ===")
plot_audio_duration(test_df, "Audio Duration Distribution (test)")

# 2. Input feature mean/std distribution
def plot_feature_mean_std(df, set_name="train"):
    means = [np.mean(iv[0]) if isinstance(iv, list) and len(iv)>0 else 0 for iv in df['input_values']]
    stds = [np.std(iv[0]) if isinstance(iv, list) and len(iv)>0 else 0 for iv in df['input_values']]
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(means, bins=30, edgecolor='black', alpha=0.7)
    plt.title(f"Input Feature Mean Distribution ({set_name})")
    plt.xlabel("Mean")
    plt.ylabel("Samples")
    plt.grid(alpha=0.3)

    plt.subplot(1,2,2)
    plt.hist(stds, bins=30, edgecolor='black', alpha=0.7)
    plt.title(f"Input Feature Std Distribution ({set_name})")
    plt.xlabel("Std")
    plt.ylabel("Samples")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\n# === Input Feature Mean/Std Distribution ===")
plot_feature_mean_std(train_df, set_name="train")
plot_feature_mean_std(test_df, set_name="test")

# 3. Label/token distribution
def plot_token_distribution(df, processor, set_name="train"):
    all_tokens = [token for label in df['labels'] for token in label]
    counter = Counter(all_tokens)
    tokens, counts = zip(*counter.most_common())
    tokens_str = [processor.tokenizer.convert_ids_to_tokens(tok) if hasattr(processor.tokenizer, 'convert_ids_to_tokens') else str(tok) for tok in tokens]
    
    plt.figure(figsize=(15, 5))
    plt.bar(tokens_str, counts)
    plt.title(f"Token Frequency Distribution ({set_name})")
    plt.xlabel("Token")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    total = sum(counts)
    print(f"[INFO] Number of tokens: {len(tokens_str)}, Most frequent token: '{tokens_str[0]}' ({counts[0]} times, {100*counts[0]/total:.2f}%)")

print("\n# === Label/Token Distribution (train) ===")
plot_token_distribution(train_df, processor, set_name="train")
print("\n# === Label/Token Distribution (test) ===")
plot_token_distribution(test_df, processor, set_name="test")

# 4. Text length distribution
def plot_text_length(df, title="Text Length Distribution (Word Count)"):
    lengths = [len(str(t).split()) for t in df['text']]
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=30, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel("Number of Words")
    plt.ylabel("Samples")
    plt.grid(alpha=0.3)
    plt.show()

print("\n# === Text Length Distribution (train) ===")
plot_text_length(train_df, title="Text Length Distribution (train)")
print("\n# === Text Length Distribution (test) ===")
plot_text_length(test_df, title="Text Length Distribution (test)")

# 5. Ground truth vs. predicted sample visualization, only if the 'pred_text' column exists
if 'pred_text' in test_df.columns:
    import random
    idxs = random.sample(range(len(test_df)), 5)
    for i in idxs:
        print(f"\n--- Sample {i} ---")
        print("Audio Path:", test_df.iloc[i]['audio_path'])
        print("Ground Truth:", test_df.iloc[i]['text'])
        print("Prediction:", test_df.iloc[i]['pred_text'])

# If prediction results are not available, this section remains unused until model inference is added

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from sklearn.manifold import TSNE

# 1. Load the model
model_dir = r"C:\Yue\ASR\project7_1_kinya-20250720T171957Z-1-001\project7_1_kinya\wav2vec2-base-kinya-final"
model = Wav2Vec2ForCTC.from_pretrained(model_dir)
model.eval()

# 2. Take the first N samples from test_df
N = 50  # Adjust N as needed for available GPU memory
sampled = test_df.sample(N, random_state=42)

hidden_feats = []
for i, row in sampled.iterrows():
    inputs = processor(row['speech'], sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.wav2vec2(**inputs, output_hidden_states=True)
    # Mean pooling on the last hidden state layer
    feat = outputs.hidden_states[-1].squeeze(0).mean(dim=0).cpu().numpy()  # [hidden_size,]
    hidden_feats.append(feat)

hidden_feats = np.stack(hidden_feats)  # [N, hidden_size]
labels = sampled['text'].tolist()

# 3. t-SNE dimensionality reduction and visualization
tsne = TSNE(n_components=2, random_state=0, perplexity=10)
feat_2d = tsne.fit_transform(hidden_feats)

plt.figure(figsize=(8,6))
plt.scatter(feat_2d[:,0], feat_2d[:,1])
for i, t in enumerate(labels):
    plt.text(feat_2d[i,0], feat_2d[i,1], str(i), fontsize=6)
plt.title("t-SNE of Wav2Vec2 hidden features (test set)")
plt.show()

row = test_df.iloc[0]
inputs = processor(row['speech'], sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits.squeeze(0).cpu().numpy()  # [T, vocab_size]
prob = torch.softmax(torch.tensor(logits), dim=-1).numpy()   # [T, vocab_size]

plt.figure(figsize=(12,5))
plt.imshow(prob.T, aspect='auto', origin='lower')
plt.colorbar()
plt.title('Token probability heatmap')
plt.xlabel('Frame')
plt.ylabel('Token index')
plt.show()

def add_noise(wave, snr_db):
    # Input: wave (np.array), snr_db (SNR in dB)
    rms_signal = np.sqrt(np.mean(wave**2))
    snr = 10**(snr_db/10)
    rms_noise = np.sqrt(rms_signal**2 / snr)
    noise = np.random.normal(0, rms_noise, wave.shape)
    return wave + noise

import jiwer

snr_list = [30, 20, 10, 5, 0, -5]  # dB, lower means noisier
wers = []
cers = []

for snr_db in snr_list:
    preds = []
    refs = []
    for i, row in test_df.sample(100, random_state=snr_db).iterrows():  # sample 100 examples only
        noisy = add_noise(np.array(row['speech']), snr_db)
        input_values = processor(noisy, sampling_rate=16000).input_values
        input_tensor = torch.tensor(input_values).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_text = processor.batch_decode(pred_ids)[0]
        preds.append(pred_text)
        refs.append(row['text'])
    wer = jiwer.wer(refs, preds)
    cer = jiwer.cer(refs, preds)
    print(f"SNR={snr_db} dB, WER={wer:.2f}, CER={cer:.2f}")
    wers.append(wer)
    cers.append(cer)

# Plot WER/CER versus SNR
plt.plot(snr_list, wers, marker='o', label="WER")
plt.plot(snr_list, cers, marker='s', label="CER")
plt.xlabel("SNR (dB)")
plt.ylabel("Error rate")
plt.title("Model Robustness vs Noise (WER/CER vs SNR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




