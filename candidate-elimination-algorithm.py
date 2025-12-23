# -------------------------------
# Candidate Elimination Algorithm
# -------------------------------

def is_consistent(hypothesis, example):
    return all(h == '?' or h == e for h, e in zip(hypothesis, example))


def more_general(h1, h2):
    return all(h1[i] == '?' or h1[i] == h2[i] for i in range(len(h1)))


def candidate_elimination(X, y):
    n = len(X[0])

    S = None
    for i in range(len(y)):
        if y[i].lower() == 'yes':
            S = X[i].copy()
            break

    if S is None:
        raise ValueError("No positive examples found. Cannot initialize S.")

    G = [['?' for _ in range(n)]]

    for i in range(len(X)):
        if y[i].lower() == 'yes': 
            for j in range(n):
                if S[j] != X[i][j]:
                    S[j] = '?'

            G = [g for g in G if is_consistent(g, X[i])]

        else: 
            new_G = []
            for g in G:
                if is_consistent(g, X[i]):
                    for j in range(n):
                        if S[j] != '?' and S[j] != X[i][j]:
                            new_h = g.copy()
                            new_h[j] = S[j]
                            if new_h not in new_G:
                                new_G.append(new_h)
                else:
                    new_G.append(g)

            G = [h for h in new_G if more_general(h, S)]

            G = [
                h for h in G
                if not any(other != h and more_general(other, h) for other in G)
            ]

        print(f"\nAfter instance {i + 1}:")
        print("S =", S)
        print("G =", G)

    return S, G


X = [
    ['Sunny', 'Warm', 'Normal', 'Strong'],
    ['Sunny', 'Warm', 'High', 'Strong'],
    ['Rainy', 'Cold', 'High', 'Strong'],
    ['Sunny', 'Warm', 'High', 'Weak']
]

y = ['Yes', 'Yes', 'No', 'Yes']


final_S, final_G = candidate_elimination(X, y)

print("\nFinal Specific Hypothesis:", final_S)
print("Final General Hypothesis:", final_G)
