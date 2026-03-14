import os
import PyPDF2
import json
import re

# 设定路径
base_dir = r"D:\LLM_learn\iovpdf"
output_file = os.path.join(base_dir, "iov_train_data.jsonl")

def batch_process_pdfs():
    dataset = []
    # 遍历文件夹下所有 pdf
    pdf_files = [f for f in os.listdir(base_dir) if f.endswith('.pdf')]
    
    for pdf_name in pdf_files:
        path = os.path.join(base_dir, pdf_name)
        print(f"正在解析: {pdf_name}")
        
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                # 简单的清洗逻辑：寻找包含资源分配逻辑的句子
                sentences = re.split(r'\.\s+', text)
                for s in sentences:
                    if any(k in s.lower() for k in ["resource", "allocation", "v2x", "delay", "mec"]):
                        dataset.append({
                            "instruction": "请分析车联网中的这项资源调度策略。",
                            "input": pdf_name, # 记录来源，方便溯源
                            "output": s.strip()
                        })

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
       # 修改前
# f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# 修改后：增加 ignore 逻辑，彻底根治非法字符
            f.write(json.dumps(entry, ensure_ascii=False).encode('utf-8', 'ignore').decode('utf-8') + '\n')
            #f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"✅ 搞定！共提取 {len(dataset)} 条数据。")

if __name__ == "__main__":
    batch_process_pdfs()