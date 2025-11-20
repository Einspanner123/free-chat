from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# llm = HuggingFacePipeline.from_model_id(
#     # model_id="Qwen/Qwen2.5-7B-Instruct",
#     model_id="Qwen/Qwen3-0.6B",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=True,
#         temperature=0.8,
#         repetition_penalty=1.03,
#         top_p=0.7,
#         top_k=40,
#         return_full_text=False,
#     ),
# )

local_model_path = "Qwen/Qwen3-0.6B"  # Transformers会自动查找缓存

# 直接使用transformers加载本地模型，避免通过from_model_id()可能的网络请求
tokenizer = AutoTokenizer.from_pretrained(
    local_model_path,
    trust_remote_code=False,  # Qwen模型需要这个参数
    local_files_only=True,  # 仅从本地加载，不尝试下载
)

model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    trust_remote_code=False,
    local_files_only=True,
    dtype="auto",  # 自动选择数据类型
    device_map="auto",  # 自动分配设备
)

# 创建pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.8,
    repetition_penalty=1.05,  # 惩罚重复
    top_p=0.7,  # 模型只考虑累积概率达到top_p的token集合
    top_k=40,  # 限制模型只考虑概率最高的k个token
    return_full_text=False,
)

# 创建LangChain包装器
llm = HuggingFacePipeline(pipeline=pipe)

chat_model = ChatHuggingFace(llm=llm)
systemMsg = SystemMessage(
    content="You are user's private ai secretary. Use Chinese words to response Chinese request, or English words to English request."
)
humanMsg = HumanMessage(
    content="我是一名智力低下的硕士生, 请你解释DDIM的扩散原理, 每一步每一个符号都必须合理解释, 并且用中文回答"
)
msg = [systemMsg, humanMsg]
response = chat_model.invoke(msg)
usage = response.usage_metadata
print(response.content)
print(usage)
