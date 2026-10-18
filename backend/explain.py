"""Occlusion-based explanation: the words that most drive the predicted class."""


def trigger_words(model, text, top_k=4, min_drop=0.01):
    probs = model.predict_proba([text])[0]
    predicted = int(probs.argmax())
    words = str(text).split()
    if not words:
        return []
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    variant_probs = model.predict_proba(variants)
    drops = [(words[i], float(probs[predicted] - variant_probs[i][predicted])) for i in range(len(words))]
    drops.sort(key=lambda item: -item[1])
    return [{"word": word, "weight": round(drop, 2)} for word, drop in drops[:top_k] if drop > min_drop]
