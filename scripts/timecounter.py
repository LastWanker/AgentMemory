import time
import torch
from transformers import AutoModel, AutoTokenizer

print("1. CUDA检查...")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA设备数: {torch.cuda.device_count()}")

print("\n2. 加载tokenizer...")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
print(f"Tokenizer加载时间: {time.time() - start:.2f}秒")

print("\n3. 加载模型到CPU...")
start = time.time()
model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
print(f"CPU加载时间: {time.time() - start:.2f}秒")

print("\n4. 移动模型到GPU...")
start = time.time()
model = model.to("cuda")
print(f"GPU转移时间: {time.time() - start:.2f}秒")

print("\n5. 测试推理...")
start = time.time()
inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs)
print(f"推理时间: {time.time() - start:.2f}秒")