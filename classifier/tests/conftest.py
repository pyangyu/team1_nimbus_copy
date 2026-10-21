import random

FAMILIES = [
    ("Translate '{w}' into French.", 20),
    ("What is {a} + {b}?", 90),
    ("A train leaves at {a}:15 and arrives at {b}:40. How long is the trip in minutes?", 400),
    ("Write a Python function that merges overlapping intervals named {w}.", 1500),
    ("Prove that n^3 - n is divisible by 6 for every integer n, then generalize to {w}.", 3000),
    ("AIME: find the number of ordered pairs (a, b) with a + b = {a} and gcd(a, b) = {b}, a, b <= 1000.", 12000),
]
WORDS = ["cat", "house", "river", "lamp", "orbit", "maple", "vector", "kettle"]


def workload(n, seed=0, scale=1.0, model="m1"):
    rng = random.Random(seed)
    for i in range(n):
        tmpl, base = rng.choice(FAMILIES)
        prompt = tmpl.format(w=rng.choice(WORDS), a=rng.randint(1, 99), b=rng.randint(1, 99))
        yield {"id": f"{model}-{seed}-{i}", "prompt": prompt, "model": model,
               "thinking_tokens": int(base * scale * rng.uniform(0.85, 1.15))}
