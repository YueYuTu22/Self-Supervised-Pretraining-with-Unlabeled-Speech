# compare_models.py
# Evaluate multiple pretrained models on FLEURS test set
# @author ileana bucur.

from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch, librosa
import jiwer

def load_audio(path):
    arr, _ = librosa.load(path, sr=16000)
    return arr

def evaluate_model(model_dir, dataset, name):
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model.eval()

    preds, refs = [], []
    for sample in dataset:
        speech = load_audio(sample["audio"]["path"])
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred = processor.batch_decode(pred_ids)[0]
        preds.append(pred)
        refs.append(sample["raw_transcription"])

    wer = jiwer.wer(refs, preds)
    cer = jiwer.cer(refs, preds)
    print(f"[{name}] WER = {wer:.3f}, CER = {cer:.3f}")
    return wer, cer

# === Config ===
models = {
    "Base (EN)": "./models/wav2vec2-base",
    "XLS-R (multi)": "./models/xlsr-53",
}

fleurs = load_dataset("google/fleurs", "kinyarwanda")["test"].select(range(50))

# === Run ===
for name, path in models.items():
    evaluate_model(path, fleurs, name)
