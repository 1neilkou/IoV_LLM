import fitz  # PyMuPDF
import json
import os
from openai import OpenAI
from tqdm import tqdm

# 1. 配置阿里云百炼 (DashScope) 的兼容接口
# 强烈建议将 API Key 设置在系统环境变量中：export DASHSCOPE_API_KEY="你的key"
# 如果为了测试方便，你也可以直接把 "你的_百炼_API_KEY" 替换成真实的 Key
api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    raise ValueError("❌ 找不到 API Key！")

# ================= 加入这行“照妖镜”代码 =================
print(f"🔍 [Debug] 当前使用的 API Key 是: '{api_key}'")
# =======================================================

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1" # 百炼的 OpenAI 兼容端点
)

def extract_text_from_pdf(pdf_path, chunk_size=800):
    """提取 PDF 文本并按固定字数切块"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    # 简单切块，防止单次请求 Token 超限
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def generate_qa_pairs(text_chunk):
    """调用百炼 Qwen 模型根据文本合成 QA 对"""
    prompt = f"""
    你是一个 6G 车联网 (V2X) 领域的数据标注专家。请阅读以下文献片段，为其生成 3 个高质量的问答对。
    要求：
    1. 提问要有深度，涉及概念解释、算网调度逻辑等。
    2. 回答要专业、详实。
    3. 必须严格按照以下 JSON 格式输出，不要输出任何代码块标记(```json)或其他废话：
    [
        {{"instruction": "问题1", "output": "回答1"}},
        {{"instruction": "问题2", "output": "回答2"}}
    ]
    
    文献片段：
    {text_chunk}
    """
    
    try:
        response = client.chat.completions.create(
            model="qwen-plus", # 使用百炼的 qwen-plus 模型，性价比极高
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,   # 降低温度，保证 JSON 格式的稳定性
            response_format={"type": "json_object"} # 强制要求返回 JSON 格式 (百炼兼容该特性)
        )
        
        # 提取并解析内容
        content = response.choices[0].message.content
        # 处理可能被包裹在 {"qa_list": [...]} 中的情况
        parsed_json = json.loads(content)
        
        # 如果模型返回的是字典包含列表，提取列表；如果是纯列表则直接返回
        if isinstance(parsed_json, dict):
            for key, value in parsed_json.items():
                if isinstance(value, list):
                    return value
            return [parsed_json]
        elif isinstance(parsed_json, list):
            return parsed_json
        else:
            return []
            
    except Exception as e:
        print(f"❌ 生成失败或 JSON 解析错误，跳过该块: {e}")
        return []

if __name__ == "__main__":
    # 请确保同级目录下有这个 PDF 文件
    pdf_file = "v2x_paper_1.pdf" 
    output_file = "v2x_domain_qa.jsonl"
    
    print("📄 1. 正在使用 PyMuPDF 解析 PDF...")
    try:
        chunks = extract_text_from_pdf(pdf_file)
    except FileNotFoundError:
        print(f"找不到文件 {pdf_file}，请检查路径！")
        exit()
    
    print(f"🧠 2. 共切分为 {len(chunks)} 块，开始调用 阿里云百炼 (qwen-plus) 合成数据...")
    all_qa = []
    
    # 建议先跑前 5 块测试一下 API 是否连通、格式是否正确
    for chunk in tqdm(chunks[:5]): 
        qa_pairs = generate_qa_pairs(chunk)
        if qa_pairs:
            all_qa.extend(qa_pairs)
        
    print(f"💾 3. 成功生成 {len(all_qa)} 条微调数据！正在保存到 {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for qa in all_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
            
    print("✅ 任务完成！")