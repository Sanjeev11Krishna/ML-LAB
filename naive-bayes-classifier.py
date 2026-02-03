import math
from transformers import pipeline

# ----------------------------
# Step 1: Dataset
# ----------------------------

satisfied = [
    "service was quick and helpful",
    "happy with customer support",
    "support team was polite",
    "good service experience",
    "problem solved quickly"
]

unsatisfied = [
    "very slow response",
    "issue not resolved",
    "poor customer care",
    "worst support ever",
    "not helpful at all"
]

# ----------------------------
# Step 2: Naïve Bayes Implementation
# ----------------------------

# Count words in each class
def word_count(feedback_list):
    counts = {}
    for feedback in feedback_list:
        for word in feedback.split():
            counts[word] = counts.get(word, 0) + 1
    return counts

satisfied_words = word_count(satisfied)
unsatisfied_words = word_count(unsatisfied)

# Prior probabilities
total_feedbacks = len(satisfied) + len(unsatisfied)
p_satisfied = len(satisfied) / total_feedbacks
p_unsatisfied = len(unsatisfied) / total_feedbacks

# Vocabulary and total words
vocabulary = set(list(satisfied_words.keys()) + list(unsatisfied_words.keys()))
V = len(vocabulary)
total_satisfied_words = sum(satisfied_words.values())
total_unsatisfied_words = sum(unsatisfied_words.values())

# Naïve Bayes prediction function
def predict_naive_bayes(feedback):
    words = feedback.split()
    satisfied_score = math.log(p_satisfied)
    unsatisfied_score = math.log(p_unsatisfied)
   
    for word in words:
        satisfied_score += math.log((satisfied_words.get(word, 0) + 1) / (total_satisfied_words + V))
        unsatisfied_score += math.log((unsatisfied_words.get(word, 0) + 1) / (total_unsatisfied_words + V))
   
    if satisfied_score > unsatisfied_score:
        return "Satisfied"
    else:
        return "Unsatisfied"

# ----------------------------
# Step 3: Test Feedbacks
# ----------------------------

test_feedbacks = [
    "quick and helpful service",
    "poor customer support",
    "support team was polite",
    "very slow response",
    "not helpful at all",
    "happy with the service"
]

print("Naïve Bayes Predictions:\n")
for feedback in test_feedbacks:
    result = predict_naive_bayes(feedback)
    print(f"Feedback: '{feedback}' -> Prediction: {result}")

# ----------------------------
# Step 4: Hugging Face Pretrained Model
# ----------------------------

# Load sentiment analysis pipeline
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


print("\nHugging Face Model Predictions:\n")
for feedback in test_feedbacks:
    result = classifier(feedback)[0]
    label = "Satisfied" if result['label'] == "POSITIVE" else "Unsatisfied"
    print(f"Feedback: '{feedback}' -> Prediction: {label} (Score: {result['score']:.2f})")

# ----------------------------
# Step 5: Comparison (Optional)
# ----------------------------

print("\nComparison Table:")
print("Feedback".ljust(35), "Naive Bayes".ljust(15), "Hugging Face")
for feedback in test_feedbacks:
    nb_result = predict_naive_bayes(feedback)
    hf_result = "Satisfied" if classifier(feedback)[0]['label'] == "POSITIVE" else "Unsatisfied"
    print(feedback.ljust(35), nb_result.ljust(15), hf_result)
