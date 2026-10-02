# 框架与 API：可以怎么讲

## Sentence-Transformers

本项目实际调用 `SentenceTransformer(model_name, device="cpu")` 加载编码器，使用 `encode(..., normalize_embeddings=True)` 编码文档和问题，通过矩阵与向量的点积做精确检索。

底层使用 PyTorch 与 Hugging Face Transformers 生态。本项目没有直接实现神经网络训练循环，所以简历中应写“使用 Sentence-Transformers 进行句向量编码”，不要扩张成“掌握分布式训练”。

使用默认 BGE 模型时，查询侧加模型推荐的检索指令；文档侧不加。换成 E5 等模型时，不能沿用当前默认约定，应按其模型卡修改 query/document 前缀。

## OpenAI Python SDK

`OpenAI(...).chat.completions.create(model=..., messages=...)` 是本项目的生成调用。使用环境变量传入凭证，并显式设置超时与有限重试。这里选择 Chat Completions 是为了覆盖常见兼容服务，实际模型 ID 由用户的服务商决定。

兼容服务可能存在参数与行为差异，因此没有硬编码 temperature 等可选参数。`OPENAI_BASE_URL` 是可选配置，未设置则使用 SDK 的默认地址。

## Transformers 本地生成：学习片段，未纳入运行验证

以下仅展示 API 顺序。模型下载、许可证、内存与具体模型 ID 需要另行确认，不是本仓库已验证的功能。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "your-compatible-chat-model"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "用一句话解释 RAG。"}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
with torch.inference_mode():
    tokens = model.generate(**inputs, max_new_tokens=80, do_sample=False)
new_tokens = tokens[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
```

`apply_chat_template` 负责角色格式，`inference_mode` 减少推理时的求导开销，只解码新增 token 可以避免把用户输入重复当成答案。
