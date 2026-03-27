import json
import argparse
from collections import Counter

def calculate_distinct_n(text, n):
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    return len(set(ngrams)) / len(ngrams)

def calculate_rep_n(text, n):
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    counts = Counter(ngrams)
    repeats = sum(count - 1 for count in counts.values())
    return repeats / len(ngrams)

def analyze_file(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    texts = []
    for line in lines:
        try:
            data = json.loads(line)
            texts.append(data.get("prediction", ""))
        except:
            pass

    if not texts:
        return None

    d2_scores = [calculate_distinct_n(t, 2) for t in texts]
    r4_scores = [calculate_rep_n(t, 4) for t in texts]

    avg_d2 = sum(d2_scores) / len(d2_scores)
    avg_r4 = sum(r4_scores) / len(r4_scores)
    
    # Also calculate AGL from text length for reference
    agl = sum(len(t.split()) for t in texts) / len(texts)

    return {
        "Distinct-2": round(avg_d2, 4),
        "Rep-4": round(avg_r4, 4),
        "Avg Word Length": round(agl, 1)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="/root/autodl-tmp/A-OSP_Project/logs/eval_results/mmhal_full_base_results.jsonl")
    parser.add_argument("--aosp_file", type=str, default="/root/autodl-tmp/A-OSP_Project/logs/eval_results/mmhal_full_aosp_results.jsonl")
    args = parser.parse_args()

    print(f"Analyzing Lexical Diversity...")
    print(f"Base file: {args.base_file}")
    res_base = analyze_file(args.base_file)
    print(f"Base Results : {res_base}")

    print(f"A-OSP file: {args.aosp_file}")
    res_aosp = analyze_file(args.aosp_file)
    print(f"A-OSP Results: {res_aosp}")

if __name__ == "__main__":
    main()
