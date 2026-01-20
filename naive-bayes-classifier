import math
from collections import defaultdict

documents = [
    ("satisfied", "service was quick and useful"),
    ("unsatisfied", "slow response"),
    ("satisfied", "happy with customer support"),
    ("unsatisfied", "issue not resolved"),
    ("satisfied", "support team was polite"),
    ("unsatisfied", "poor customer care"),
    ("satisfied", "good service experience"),
    ("unsatisfied", "worst support experience"),
    ("satisfied", "problem solved quickly"),
    ("unsatisfied", "not helpful at all")
]

word_count = defaultdict(lambda: defaultdict(int))
class_count = defaultdict(int)
vocab = set()

for label, text in documents:
    class_count[label] += 1
    words = text.split()
    for word in words:
        word_count[label][word] += 1
        vocab.add(word)

def predict(text):
    words = text.split()
    scores = {}

    for label in class_count:
        scores[label] = math.log(class_count[label] / sum(class_count.values()))
        
        for word in words:
            scores[label] += math.log(
                (word_count[label][word] + 1) /
                (sum(word_count[label].values()) + len(vocab))
            )

    return max(scores, key=scores.get)

test_text = "quick and helpful service"
print("Prediction:", predict(test_text))
