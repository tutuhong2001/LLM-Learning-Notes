# Hugging Face Transformers 推理

Transformers 常用 AutoTokenizer 加载分词器，AutoModelForCausalLM 加载自回归语言模型。对聊天模型应使用模型适配的 chat template，避免手写错误的角色格式。

model.eval() 设置评估模式，torch.inference_mode() 关闭自动求导相关开销。generate 的 max_new_tokens 控制新增 token 上限，do_sample 决定是否采样；temperature 通常在启用采样时影响输出随机性。

首次使用要下载权重，显存或内存需求取决于参数量、精度和上下文长度。半精度与量化可能降低内存需求，但取决于硬件和库的支持。本仓库实际运行的是句向量编码器；没有验证本地生成式大模型推理。
