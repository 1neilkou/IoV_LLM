# analyze_rm_scores.py
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

def score_texts(model, tokenizer, texts, max_length=1024):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits

    return logits.float().view(-1).cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="./models/Qwen2.5-7B-Instruct")
    parser.add_argument("--rm_adapter_path", type=str, default="./output/iov_qwen_rm/final_rm")
    parser.add_argument("--data_file", type=str, default="data/v2x_rm_preference.jsonl")
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, args.rm_adapter_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    with open(args.data_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    data = data[:args.max_samples]

    chosen_better = 0
    margins = []
    examples = []

    for item in tqdm(data, desc="Analyzing RM scores"):
        chosen_text = item["prompt"] + "\n\n" + item["chosen"]
        rejected_text = item["prompt"] + "\n\n" + item["rejected"]

        scores = score_texts(model, tokenizer, [chosen_text, rejected_text])
        chosen_score, rejected_score = scores[0], scores[1]
        margin = chosen_score - rejected_score

        if chosen_score > rejected_score:
            chosen_better += 1

        margins.append(margin)

        if len(examples) < 5:
            examples.append({
                "prompt": item["prompt"],
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "margin": margin,
                "chosen": item["chosen"],
                "rejected": item["rejected"]
            })

    acc = chosen_better / len(data) if data else 0.0
    avg_margin = sum(margins) / len(margins) if margins else 0.0

    result = {
        "num_samples": len(data),
        "pairwise_accuracy": acc,
        "avg_margin": avg_margin,
        "example_cases": examples
    }

    print("✅ RM 分析完成")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    with open("output/rm_score_analysis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()