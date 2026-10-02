# OpenAI 兼容接口

Python OpenAI SDK 可通过 OpenAI 客户端调用 chat.completions.create，传入 model 和 messages 获取回答。部分服务提供兼容接口，通过 base_url 指定服务地址。

OPENAI_API_KEY 用于认证，OPENAI_MODEL 指定服务商实际支持的模型 ID，OPENAI_BASE_URL 可选。兼容协议不保证支持所有参数，应查阅具体服务商文档。不要把密钥写入源码或提交到 Git。

网络请求应设置超时和有限重试，注意速率限制与调用费用。本项目的 ask 命令会发送问题以及检索到的文档给配置的服务商；--dry-run 只在本地打印请求，不会调用 API。
