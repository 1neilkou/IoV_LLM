import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# 1. 设置本地模型路径
model_id = "./models/Qwen2.5-7B-Instruct"

# 2. 设置本地数据集路径
dataset_path = "./data/iov_train_data.jsonl"

# 3. 强制开启离线模式，防止模型再次尝试联网
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

def main():
    print("正在初始化5090实验环境")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,   
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    print("\nLoRA注入成功！5090显存当前占用： {: .2f} GB".format(torch.cuda.memory_allocated() / 1024**3
    ))

if __name__ == "__main__":
    main()  


 



# def get_lora_model(rank):
#     model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
#     config = LoraConfig(
#         r=rank,
#         target_modules=["q_proj",  "v_proj"],
#         task_type=TaskType.CAUSAL_LM
#     )
#     lora_model = get_peft_model(model, config)
#     lora_model.print_trainable_parameters()
#     return lora_model

# print("---运行Rank 8 实验---")
# get_lora_model(8)
# print("---运行Rank 64 实验---")
# get_lora_model(64)


  