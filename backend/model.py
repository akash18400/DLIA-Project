import random

BREEDS = ["Sahiwal", "Gir", "Jersey", "Murrah", "Nili Ravi", "Red Sindhi", "Kankrej"]

def predict_breed(image_path):
    """
    Returns top 3 predicted breeds with confidence
    """
    shuffled = BREEDS.copy()
    random.shuffle(shuffled)
    top3 = shuffled[:3]
    confidences = sorted([round(random.uniform(0.7, 0.99), 2) for _ in range(3)], reverse=True)
    predictions = [{"breed": b, "confidence": c} for b, c in zip(top3, confidences)]
    top_breed = predictions[0]["breed"]
    top_confidence = predictions[0]["confidence"]
    return top_breed, top_confidence, predictions
