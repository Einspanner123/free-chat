# Context Management System — 实现规划

> **目的：** 将当前固定的"取 10 条消息"升级为生产级上下文管理系统，包含 token 预算、智能压缩、话题选择和推理流水线并发。

---

## 一、设计决策（回答你的问题）

### 1.1 Token 计数：是否准确？是否必要？如何设计？

**必要性：是，但精度分两档。**

| 用途 | 精度要求 | 方案 |
|------|---------|------|
| 上下文装配决策（预算计算） | ±5% 足够 | 轻量估算器 |
| 日志/监控/计费 | 精确 | 模型 tokenizer 精确计数 |

**决策：不要在 Go 侧实现精确 tokenizer。** Go 生态中没有能覆盖所有 HuggingFace 模型的 tokenizer（`tiktoken-go` 只覆盖 OpenAI/Llama 系列）。准确计数必须走 Python。

**架构方案 — Hybrid 双层计数：**

```
Go 侧 (快速预算估算)
  └─ 使用 tiktoken-go (支持 GPT/Llama/Qwen tokenizer)
  └─ 误差 ±3-5%，足够做滑动窗口决策
  └─ 退化：tiktoken-go 不支持模型时 ≈ len(text)/2 (中文安全估算)

Python 侧 (精确计数，存 Redis)
  └─ tokenizer.encode(msg) 精确 token 数
  └─ 每条消息持久化时附带 token_count 字段
  └─ Go 读取时直接取 token_count，无需重复计算
```

**新字段设计：**
```go
// domain/entity.go
type Message struct {
    ID         string
    SessionID  string
    UserID     string
    Role       Role
    Content    string
    TokenCount int       // ← 新增：Python 端写入时精确计算
    CreatedAt  time.Time
}
```

存储时 Python 端计算并回写 `token_count`，Go 端只管读取。冷启动（首次消息未入库时）用 Go 端的 `tiktoken-go` 估算。

---

### 1.2 上下文裁剪策略：两个方案

#### 方案 A：话题分类 + 用户选择

**核心思路：** LLM 分析对话历史，聚类出话题分支，前端展示让用户选。

```
触发条件：Context Builder 发现预算不足时
  ↓
1. 提取历史消息（从旧到新）
  ↓
2. 调用 LLM（Python 侧 /chat/analyze 端点）：
   "分析以下对话，识别出讨论过的话题。
    每个话题包含：标签(5字内)、一句话摘要、涉及的消息ID范围"
  ↓
3. Python 返回话题列表：
   [
     {"label":"微服务架构","summary":"讨论了DDD和gRPC","msg_ids":[1-15]},
     {"label":"部署方案","summary":"对比了Docker Compose和K8s","msg_ids":[16-30]},
     {"label":"当前问题","summary":"正在调试RocketMQ连接","msg_ids":[31-40]}
   ]
  ↓
4. Go 端将话题列表塞入 SSE 特殊事件，前端弹选择器
  ↓
5. 用户选中话题 → 前端发送新请求含 topic_id
  ↓
6. Context Builder 只保留该话题范围内消息 + 就近 3 条缓冲
```

**代价分析：**
- 每次分析 ≈ 会话总 token 数的 1.5x（输入+输出）
- 不应每次触发，而应在预算连续 2 次不足时触发
- 分析结果缓存到 Redis（ttl=1h），避免重复分析

#### 方案 B：自动压缩（无需用户介入）

**核心思路：** 分层压缩，越旧越粗略，不丢失语义。

```
Level 0 (最近 5 轮)          ── 原文保留
Level 1 (第 6-20 轮)         ── 每条压缩为单句摘要
Level 2 (第 21-50 轮)        ── 按话题合并为段落摘要
Level 3 (50 轮之前的会话)     ── 标题级关键词
Level 4 (预算仍不足)          ── 删除（存数据库可查询）
```

**压缩时机：** 异步。每次 SaveAssistantMessage 后检查预算，若不足则触发压缩 goroutine。

**压缩格式示例：**
```json
// Level 1 压缩
{"role":"compressed","content":"用户询问了gRPC流式传输的三种模式","original_tokens":340,"compressed_tokens":18}

// Level 2 压缩
{"role":"compressed","content":"话题「微服务通信」: 讨论了gRPC vs REST的性能差异、ProtoBuf schema设计、流式传输模式选择","original_tokens":1200,"compressed_tokens":48}
```

#### 混合策略

```
每次请求时：
  ContextBuilder.Build():
    1. 获取历史消息
    2. 计算总 token 数
    3. if budget_ok → 全量返回
    4. if 轻微超预算 → 方案 B (自动压缩, 静默)
    5. if 严重超预算 → 方案 A (话题选择, 交互)
    6. if 极端超预算 → 方案 B Level 4 丢弃 + 方案 A
```

---

### 1.3 Attention Sink 因素

**现象**（"Efficient Streaming LLM with Attention Sinks", 2023）：
- Softmax attention 必须在所有位置上分配概率质量
- 初始 token 无论内容如何都会吸收异常多的 attention
- 这导致位置 0 附近的 token 语义被"淹没"

**对应设计：**

```
正确的前缀结构：

[SINK_TOKEN]            ← 一个空 token（如 \n\n），专门吸收 attention sink
[SYSTEM_PROMPT]         ← 系统指令（位置 1-3，sink 效应已减弱）
[GLOBAL_INSTRUCTIONS]   ← 全局行为指令
---
[SESSION_SUMMARY]       ← 可选：压缩后的会话摘要
[COMPRESSED_HISTORY]    ← 可选：方案 B 的压缩内容
---
[RECENT_MESSAGES]       ← 最近 N 条（原文保留）
[CURRENT_INPUT]         ← 当前用户输入
[GENERATION]            ← 模型输出
```

**关键点：**
- 不要将关键指令放在位置 0，放在 sink token 之后
- 重复关键指令在上下文的开头和结尾（首位效应 + 近因效应）
- Qwen 系列的 `apply_chat_template` 已经隐式处理了 sink 结构（它会在 system 前加 `<|im_start|>`），但上层仍要注意
- **StreamingLLM 备选**：如果未来切换到 KV-cache 优化的推理引擎，可以显式插入 sink token

---

### 1.4 前缀优化提高命中率

**前缀命中率**指模型在生成时正确关注上下文关键信息的能力。

**策略：**

1. **关键信息重复**：最重要的指令（如"用中文回答"）出现在 system prompt 末尾 + context 末尾 → 首位+近因双保险
2. **密度优先**：system prompt 控制在 200 token 以内，拆成：
   ```
   [固定角色定义] → [任务相关的一次性指令(用户不可见)] → [用户设定的自定义指令]
   ```
3. **衰减层**：越靠前的内容自动获得 attention 衰减（通过引入注意力掩码或结构化前缀），但当前 transformers 原生不支持——通过格式化的隐性手段：空行分隔段落
4. **KV Cache 复用**（未来方向）：如果 system prompt + global instructions 固定不变，可以用 vLLM 的 prefix caching 跳过它们的 prefill 阶段

---

### 1.5 我的补充方向

1. **Token 预算分层配置**：
   ```
   max_context_window = model_max_tokens - reserved_output - safety_margin
   reserved_output = max_new_tokens × 1.2
   safety_margin = 256
   ```

2. **压缩质量监控**：
   - 每次压缩后计算 `compression_ratio = compressed_tokens / original_tokens`
   - 如果 ratio > 0.5（压缩效果差），标记该会话"不易压缩"，下次直接走方案 A
   - 周期性评估：压缩后的对话质量是否下降（对比用户满意度指标）

3. **渐进式加载**（UX 优化）：
   - 首次请求：先发近 3 条消息（低延迟）
   - 在模型生成的同时，异步加载剩余上下文
   - 如果在首个 token 返回前上下文就绪，替换为完整上下文
   - 这利用了 TTFT（Time To First Token）和首 token 之后生成时间的间隙

4. **支持模型动态配置**：
   - 不同模型有不同 max_tokens（Qwen3-0.6B=32K, Qwen3-3B=32K, DeepSeek=128K）
   - `model_max_tokens` 从服务注册/配置文件读取，不硬编码

---

## 二、实现任务拆解

### Phase 1：基础设施（3 个任务）

---

#### Task 1.1：Message 模型增加 TokenCount 字段

**Objective:** 让每条消息携带 token 数，为预算计算打基础

**Files:**
- Modify: `services/chat-service/internal/domain/entity.go`
- Modify: `services/chat-service/internal/infrastructure/persistence/model/message_model.go`
- Modify: `services/chat-service/internal/infrastructure/persistence/cache/session_message_cache.go`
- Modify: `services/chat-service/internal/infrastructure/persistence/repository/message_repository.go`
- Test: `services/chat-service/internal/domain/entity_test.go`

**Changes:**
```go
// entity.go
type Message struct {
    ID         string
    SessionID  string
    UserID     string
    Role       Role
    Content    string
    TokenCount int       // 新增
    CreatedAt  time.Time
}
```

Proto 文件也要加 field，但为了避免改 proto 重编译，这次只走 Go 内部。下游存储（Redis/DB）反序列化时 token_count=0 兼容旧数据。

**Verify:** `go build ./... && go test ./...`

---

#### Task 1.2：Python 端精确 token 计数

**Objective:** 消息入库时 Python 侧计算 token 数并回写

**Files:**
- Modify: `services/llm-inference/src/chat_model.py`
- Modify: `services/llm-inference/src/server.py`
- Test: `services/llm-inference/src/test_client.py` (手动验证)

**Changes:**
```python
# chat_model.py — 新增 count_tokens 方法
def count_tokens(self, text: str) -> int:
    return len(self.tokenizer.encode(text))

# server.py — StreamInference 中回写 token_count
# 在 yield InferenceResponse 时填充 generated_tokens 字段
# 已经做了（resp.generated_tokens = gen_tokens）
```

这块已经部分实现了（`self.tokenizer.tokenize(chunk)`），但不精确（`tokenize` 返回的是字符级别）。改为 `self.tokenizer.encode(text)` 得到准确的 token IDs 数量。

**Verify:** 手动运行 `python src/test_client.py` 检查日志中的 token 数

---

#### Task 1.3：Go 侧 tiktoken-go 轻量估算

**Objective:** Go 端在 Python 尚未写入 token_count 时有后备估算

**Files:**
- Create: `services/chat-service/internal/infrastructure/tokenizer/tokenizer.go`
- Test: `services/chat-service/internal/infrastructure/tokenizer/tokenizer_test.go`

**Changes:**
```go
package tokenizer

type Tokenizer interface {
    Count(text string) int
    Model() string
}

func NewTokenizer(modelName string) Tokenizer {
    // 使用 tiktoken-go，根据 modelName 选择编码
    // 退化：len(text)/2（中文安全值）
}
```

**Dependency:** 添加 `github.com/pkoukk/tiktoken-go`

**Verify:** `go test ./services/chat-service/internal/infrastructure/tokenizer/... -v`

---

### Phase 2：Context Builder 核心（4 个任务）

---

#### Task 2.1：ContextBuilder 接口 + Budget 计算

**Objective:** 定义上下文装配的核心接口和预算算法

**Files:**
- Create: `services/chat-service/internal/domain/context.go`
- Create: `services/chat-service/internal/infrastructure/context/builder.go`
- Create: `services/chat-service/internal/infrastructure/context/budget.go`
- Test: `services/chat-service/internal/infrastructure/context/budget_test.go`

**Budget 算法：**
```go
type Budget struct {
    MaxContextWindow int // 从模型配置读取
    ReservedOutput   int // max_new_tokens * 1.2
    SafetyMargin     int // 256
    UsedTokens       int // 已用 token 数
}

func (b *Budget) Available() int {
    return b.MaxContextWindow - b.ReservedOutput - b.SafetyMargin - b.UsedTokens
}

func (b *Budget) IsExhausted() bool {
    return b.Available() <= 0
}
```

**ContextBuilder 接口：**
```go
type ContextBuilder interface {
    Build(ctx context.Context, sessionID, userID, userMessage string) (*BuiltContext, error)
}

type BuiltContext struct {
    Messages    []*domain.Message  // 最终上下文消息列表
    Strategy    string             // "full" | "compressed" | "topic_select"
    Compression map[string]any     // 压缩元数据（调试/监控用）
}
```

**Verify:** `go test ./services/chat-service/internal/infrastructure/context/... -v`

---

#### Task 2.2：全量模式 + 自动压缩模式（方案 B）

**Objective:** 实现方案 B 的静默压缩

**Files:**
- Create: `services/chat-service/internal/infrastructure/context/compressor.go`
- Modify: `services/chat-service/internal/infrastructure/context/builder.go`
- Test: `services/chat-service/internal/infrastructure/context/compressor_test.go`

**Compressor 接口：**
```go
// 向 Python 侧发压缩请求
type Compressor interface {
    Compress(ctx context.Context, sessionID string, messages []*domain.Message, targetBudget int) ([]*CompressedSegment, error)
}

type CompressedSegment struct {
    OriginalTokens  int
    CompressedTokens int
    Content         string  // 压缩后的摘要
    Role            string  // "compressed"
    Level           int     // 0=原文, 1=单句, 2=段落, 3=标题
}
```

**实现方式：** Go 端调用 Python 侧的 `/compress` gRPC 端点（Python 端用 LLM 做摘要压缩）。

**Python 端新增端点：**
```python
# server.py
def CompressContext(self, request, context):
    """接收消息列表，返回压缩后的摘要"""
    pass
```

**注意：** 新增 gRPC 端点需要改 proto 文件重新编译。折中方案：
- 在本阶段先不做跨 gRPC 调用
- Compressor 在 Go 端用启发式方式实现（按长度裁切，不做语义压缩）
- 语义压缩放到 Phase 3

**Verify:** `go test ... -v`

---

#### Task 2.3：话题分类模式（方案 A）

**Objective:** 实现方案 A 的话题聚类 + 前端选择

**Files:**
- Create: `services/chat-service/internal/infrastructure/context/topic_analyzer.go`
- Modify: `services/chat-service/internal/infrastructure/context/builder.go`
- Modify: `services/api-gateway/internal/handler/chat.go` (SSE 事件增加 topic 事件)
- Modify: `services/chat-service/internal/domain/context.go`
- Test: `services/chat-service/internal/infrastructure/context/topic_analyzer_test.go`

**TopicAnalyzer 接口：**
```go
type TopicAnalyzer interface {
    Analyze(ctx context.Context, sessionID string) ([]*Topic, error)
}

type Topic struct {
    ID       string   `json:"id"`
    Label    string   `json:"label"`     // 5字内标签
    Summary  string   `json:"summary"`   // 一句话摘要
    MsgRange [2]int   `json:"msg_range"` // 消息 ID 范围
}
```

**SSE 事件类型新增：**
```
event: topic_select
data: {"topics": [{"id":"...", "label":"微服务架构", ...}], "reason":"context_overflow"}

event: topic_selected  
data: {"topic_id": "..."}  ← 用户选择后发送
```

**Go → Python 话题分析调用：**
- 新增 Python gRPC 端点 `AnalyzeTopics`
- 或通过聊天推理接口间接实现（将历史作为输入，让 LLM 输出结构化 JSON）

**Verify:** 集成测试覆盖 SSE 事件序列

---

#### Task 2.4：集成 ContextBuilder 到 StreamChat 主流程

**Objective:** 用 ContextBuilder 替换当前的 `GetContext()`

**Files:**
- Modify: `services/chat-service/internal/interfaces/chat.go`
- Modify: `services/chat-service/cmd/main.go` (DI 注入 ContextBuilder)

**Changes:**
```go
// chat.go — StreamChat 中
// 替换：
//   contextStr, err := h.app.GetContext(ctx, sessionID)
// 为：
builtCtx, err := h.ctxBuilder.Build(ctx, sessionID, req.UserId, req.Message)
if err != nil {
    log.Printf("[WARN] context build failed, falling back to plain message: %v", err)
    inferenceReq.Request = req.Message
} else {
    // builtCtx.Messages → JSON → inferenceReq.Request
    jsonBytes, _ := json.Marshal(builtCtx.Messages)
    inferenceReq.Request = string(jsonBytes)
    
    // 如果方案 A 触发了话题选择，在 SSE 中发送 topic_select 事件
    if builtCtx.Strategy == "topic_select" && buildCtx.Topics != nil {
        // 发送话题选择事件，然后等待用户选择
    }
}
```

**Verify:** 全量集成测试

---

### Phase 3：Attention Sink + 前缀优化（2 个任务）

#### Task 3.1：前缀结构优化

**Objective:** 按 attention sink 原则重组消息顺序

**Files:**
- Modify: `services/chat-service/internal/infrastructure/context/builder.go`
- Test: 在现有 builder test 中新增

**实现：**
```go
// builder.go
func (b *ContextBuilder) assemble(messages []*domain.Message, userID, userMessage string) []*domain.Message {
    var result []*domain.Message
    
    // 1. Sink token (空消息, 仅用于 attention sink)
    result = append(result, &domain.Message{Role: domain.RoleSystem, Content: "\n\n"})
    
    // 2. System prompt
    result = append(result, &domain.Message{Role: domain.RoleSystem, Content: b.systemPrompt})
    
    // 3. 用户画像/偏好 (可选)
    // 4. 会话摘要 (压缩后的历史)
    // 5. 最近对话
    result = append(result, messages...)
    
    // 6. 当前用户输入
    result = append(result, &domain.Message{Role: domain.RoleUser, Content: userMessage})
    
    return result
}
```

---

#### Task 3.2：关键指令重复策略

**Objective:** 重要指令出现在开头和结尾

**Files:**
- Modify: `services/chat-service/internal/infrastructure/context/builder.go`

**实现：**
```go
systemPrompt := `You are a helpful assistant.
Always respond in Chinese unless the user writes in another language.
Keep responses concise and accurate.`

// 首部完整
// 尾部重申关键约束
suffixConstraint := &domain.Message{Role: domain.RoleSystem, Content: "Remember: respond concisely and accurately."}
```

---

### Phase 4：流水线并发（1 个任务）

#### Task 4.1：推理后端升级为 vLLM

**Objective:** 用 vLLM 替换 transformers，获得 continuous batching

**Files:**
- Modify: `services/llm-inference/pyproject.toml`
- Modify: `services/llm-inference/src/server.py`
- Rewrite: `services/llm-inference/src/chat_model.py` → 使用 `vllm.AsyncLLMEngine`
- Rewrite: `services/llm-inference/src/config.py` (vLLM 参数)

**架构：**
```python
# vLLM 版本
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model=model_path,
    max_model_len=32768,
    gpu_memory_utilization=0.9,
))

async def stream_inference(messages_text):
    # vLLM 原生支持流式 + 并发
    async for result in engine.generate(messages_text, sampling_params):
        yield result
```

**并发模型对比：**

| 方案 | 吞吐 (req/s) | 实现复杂度 |
|------|-------------|-----------|
| 当前 Thread+Lock | ~0.3 | 低 |
| vLLM Continuous Batching | ~5-10 | 中 |
| TGI | ~3-5 | 中 |

**迁移注意：**
- vLLM 要求特定版本的 CUDA/PyTorch
- 需要更新 Dockerfile（nvidia/cuda:12.x）
- 评估当前 GPU 型号兼容性（A6000 ✅）

---

## 三、测试策略

| 层级 | 覆盖内容 | 工具 |
|------|---------|------|
| 单元测试 | Tokenizer、Budget、ContextBuilder 各模式 | Go testing |
| 单元测试 | Python tokenizer count | pytest |
| 集成测试 | 完整上下文装配流（全量→压缩→话题选择） | Go test + miniredis |
| 端到端测试 | SSE 事件序列 topic_select/topic_selected | 手动 + test_client.py |
| 性能测试 | ContextBuilder 在不同历史长度下的延迟 | Go benchmark |

---

## 四、执行顺序总结

```
Phase 1: 基础设施
  Task 1.1 Message.TokenCount 字段  ──→ 提交
  Task 1.2 Python 端精确计数       ──→ 提交
  Task 1.3 Go 端 tiktoken-go 估算  ──→ 提交

Phase 2: Context Builder
  Task 2.1 接口 + Budget           ──→ 提交
  Task 2.2 自动压缩（方案 B）       ──→ 提交
  Task 2.3 话题分类（方案 A）       ──→ 提交
  Task 2.4 集成到 StreamChat       ──→ 提交

Phase 3: Attention Sink + 前缀优化
  Task 3.1 前缀结构优化             ──→ 提交
  Task 3.2 关键指令重复             ──→ 提交

Phase 4: 推理流水线
  Task 4.1 vLLM 迁移               ──→ 提交
```

## 五、未解决问题（待讨论）

1. **Topic 分析的触发频率？** 每次预算不足都分析太贵。建议：连续 2 次 Build 走 Level 4(丢弃)时才触发
2. **方案 A 的 Topic 分析 LLM 调用用大模型还是小模型？** 如果用小模型（0.6B）做分析，质量够吗？还是复用同一个推理服务？
3. **vLLM 迁移的 GPU 兼容性？** 当前 Dockerfile 基于 nvidia/cuda:11.8，vLLM 需要 ≥12.1。需要升级基础镜像
4. **前端 topic 选择器的 UI 方案？** 纯 SSE 交互（推 + 等响应）在 HTTP 长连接下是否可行，还是需要 WebSocket？
