# error_analysis.py
# Analyze common recognition errors from ASR predictions
# @author ileana bucur.

import jiwer
from collections import Counter
import difflib

def most_common_word_errors(refs, preds, top_n=10):
    substitutions = jiwer.ProcessedTruthHypothesis(
        truth=refs, hypothesis=preds, truth_aligned=[], hypothesis_aligned=[], steps=[]
    )
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(), 
        jiwer.RemovePunctuation(), 
        jiwer.RemoveWhiteSpace(replace_by_space=True), 
        jiwer.Strip()
    ])
    measures = jiwer.compute_measures(refs, preds, truth_transform=transformation, hypothesis_transform=transformation)
    print("[INFO] Full error report:")
    print(measures)

    # Tokenize and analyze with difflib
    word_errors = []
    for r, p in zip(refs, preds):
        ref_words = r.lower().split()
        pred_words = p.lower().split()
        diff = difflib.ndiff(ref_words, pred_words)
        for d in diff:
            if d.startswith("- "):
                word_errors.append(d[2:])
    counts = Counter(word_errors)
    print(f"Top {top_n} most commonly missed words:")
    for word, freq in counts.most_common(top_n):
        print(f"{word}: {freq}")

# Example usage
if __name__ == "__main__":
    refs = ["uyu mwana aravuga neza", "mbega inkuru nziza", "iki ni igisubizo cyiza"]
    preds = ["uyu aravuga neza", "mbega inkuru", "ni igisubizo"]
    most_common_word_errors(refs, preds)
