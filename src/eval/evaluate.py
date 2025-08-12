import json, re, requests
from collections import Counter

def normalize(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9á-úà-ùâ-ûãõç ]", " ", s)
    return " ".join(s.split())

def f1(pred, gold):
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p or not g: return 0.0
    inter = sum((Counter(p) & Counter(g)).values())
    if inter == 0: return 0.0
    precision = inter / len(p)
    recall = inter / len(g)
    return 2*precision*recall/(precision+recall)

def exact_match(pred, gold):
    return 1.0 if normalize(pred) == normalize(gold) else 0.0

def main():
    qs, ems, f1s = 0, [], []
    with open("src/eval/dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qs += 1
            r = requests.post("http://localhost:8000/ask", json={"question": obj["question"]}).json()
            pred = r["answer"]
            ems.append(exact_match(pred, obj["answer"]))
            f1s.append(f1(pred, obj["answer"]))
    print(f"Q: {qs} | EM: {sum(ems)/qs:.3f} | F1: {sum(f1s)/qs:.3f}")

if __name__ == "__main__":
    main()