import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. 路径配置
BASE_MODEL = "./models/Qwen2.5-7B-Instruct"
SFT_PATH = "./output/iov_qwen_lora/final_model"
GRPO_PATH = "./output/iov_qwen_grpo/final_agent"

# 2. 模拟沙盒环境 (物理 Ground Truth)
def get_physical_truth(task_size, cpu_cycles, bandwidth):
    # 模拟物理公式计算得分
    lat_local = (cpu_cycles / 1.2e9) * 1000
    lat_edge = (task_size / bandwidth + cpu_cycles / 8.0e9) * 1000
    return "边缘卸载" if lat_edge < lat_local else "本地计算"

def run_eval(model_path, name, scenarios, tokenizer):
    print(f"\n评估模型: {name} ...")
    # 加载适配器
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()

    results = []
    for i, sc in enumerate(scenarios):
        prompt = f"任务数据量: {sc['size']}MB, 计算量: {sc['cycles']}G cycles, 带宽: {sc['bw']}MB/s。请决策并说明理由。"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128)
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        
        truth = get_physical_truth(sc['size'], sc['cycles']*1e9, sc['bw'])
        pred = "边缘卸载" if "边缘" in response or "卸载" in response else "本地计算"
        is_correct = (pred == truth)
        results.append(is_correct)
        print(f"场景 {i+1}: 真实最优={truth} | 模型决策={pred} | {'✅' if is_correct else '❌'}")
    
    # 清理显存以便加载下一个模型
    del model, base
    torch.cuda.empty_cache()
    return sum(results) / len(results)

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    scenarios = [
        {"size": 5.0, "cycles": 2.0, "bw": 2.0},  # 极低带宽 -> 应选本地
        {"size": 1.0, "cycles": 5.0, "bw": 20.0}, # 高带宽+大计算 -> 应选边缘
        {"size": 4.0, "cycles": 0.5, "bw": 10.0}, # 小计算 -> 应选本地
        {"size": 2.0, "cycles": 3.0, "bw": 5.0},  # 中等情况
        {"size": 0.5, "cycles": 10.0, "bw": 15.0} # 极端计算量 -> 必选边缘
    ]
    
    sft_acc = run_eval(SFT_PATH, "SFT Model", scenarios, tokenizer)
    grpo_acc = run_eval(GRPO_PATH, "GRPO Agent", scenarios, tokenizer)
    
    print("\n" + "="*30)
    print(f"📊 最终实战准确率对比:")
    print(f"SFT 基线模型: {sft_acc*100:.1f}%")
    print(f"GRPO 进化版: {grpo_acc*100:.1f}%")
    print("="*30)

if __name__ == "__main__":
    main()