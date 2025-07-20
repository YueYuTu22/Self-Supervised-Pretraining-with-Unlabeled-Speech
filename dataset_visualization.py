import os
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# === Path Setup ===
base_dir = r"C:/Yue/ASR/dataset/kinyarwanda-tts-dataset"
csv_path = os.path.join(base_dir, "tts-dataset.csv")
audio_dir = os.path.join(base_dir, "audio")

# === 1. Read and split the data ===
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
        data.append([file_id, text])

df = pd.DataFrame(data, columns=["file", "text"])

# === 2. Construct full audio file path ===
def extract_wav_name(file_id):
    # TTS_1_2 → TTS 1_2.wav
    if file_id.startswith('TTS_'):
        return 'TTS ' + file_id[4:] + '.wav'
    return file_id + '.wav'

df["audio_path"] = df["file"].apply(lambda x: os.path.join(audio_dir, extract_wav_name(x)))

print("Data preview:")
print(df.head())

# === 3. Visualize waveforms and spectrograms for first 3 samples ===
for i, row in df.head(3).iterrows():
    audio_path = row["audio_path"]
    text = row["text"]
    print(f"Sample {i} path: {audio_path}")
    if not isinstance(audio_path, str) or not os.path.exists(audio_path):
        print("  File does not exist or invalid path, skipping")
        continue
    print("  File exists, plotting...")
    try:
        y, sr = librosa.load(audio_path, sr=None)
        plt.figure(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr)
        plt.title(f"Waveform - {text[:50]}")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Waveform plot failed: {e}")

    try:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        plt.figure(figsize=(10, 3))
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Spectrogram - {text[:50]}")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Spectrogram plot failed: {e}")

# === 4. Text length distribution analysis ===
texts = [t for t in df["text"] if isinstance(t, str) and t.strip()]
lengths = [len(t.split()) for t in texts]

plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=20, edgecolor='black', alpha=0.7)
plt.title("Text Length Distribution (Word Count)")
plt.xlabel("Number of Words")
plt.ylabel("Number of Samples")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("First 10 text samples:", texts[:10])
print("First 10 lengths:", lengths[:10])
print("Valid sample count:", len(texts))

print("Total text samples:", len(texts))
if lengths:
    print("Mean word count:", np.mean(lengths))
    print("Shortest:", np.min(lengths), "words")
    print("Longest:", np.max(lengths), "words")
else:
    print("No valid text data for statistics!")
print("Analysis complete!")



