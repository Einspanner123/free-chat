## 目标
- 提供一个稳定的聊天后端：Go 微服务网关 + Go 聊天服务（会话与历史）+ Python LLM 推理服务。
- 将请求按模型名负载均衡到多个同名 Python 推理实例，并保持会话粘性与历史上下文。
- 具备失败重试、故障转移、限流与可观测性；容器化便于本地与生产部署。

## 架构改造
1. 网关（Go）：仅负责认证与将聊天流量代理到 Chat-Service 的 gRPC；统一入口、鉴权、限流。
2. 聊天服务（Go）：负责会话与历史存储（Redis）、粘性路由与负载均衡、将完整上下文转发到 Python 推理服务（gRPC 流）。
3. 推理服务（Python）：实现 `StreamInference`，支持多实例以同名模型注册到 Consul，返回流式 Token。
4. 服务发现：Consul 提供模型名到实例列表；服务端健康检查与去除不健康实例。

## 基础修复
- 端口规划（ENV）：`GATEWAY=8080`，`CHAT_SERVICE=8081`，`AUTH_GRPC=8082`，`AUTH_HTTP_HEALTH=8083`；所有服务按 ENV 加载，避免冲突（修 `shared/config/config.go` 引用和默认值）。
- gRPC 客户端修正：用 `grpc.Dial(host:port, grpc.WithTransportCredentials(insecure.NewCredentials()))`，移除 `http://` 前缀；为实例对象新增 `GetAddressPort()` 返回裸地址（修 `shared/registry/service_manager.go` 与调用方）。
- 入口整理：网关路由只保留需要的 REST，认证与聊天 handler 参数签名统一并传入 `ServiceManager` 与目标服务名（修 `cmd/gateway/main.go` 与 `internal/handler/*`）。
- Auth-Service 端口拆分：gRPC 独占 `AUTH_GRPC`，HTTP 仅健康检查在 `AUTH_HTTP_HEALTH` 或移除 Gin（修 `cmd/auth-service/main.go`）。
- 数据库驱动：统一选择 `pgx` 或 `lib/pq`，删除重复的 `gorm`/`database/sql` 双栈，保留一套实现并在启动时进行连接校验（修 `cmd/auth-service/internal/*`）。

## 会话与历史
- Redis 键模式保持现状：`session:<id>`、`session_messages:<session_id>`、`user_sessions:<user_id>`。
- 新增粘性映射键：`session_affinity:<session_id> = <instance_id>`，当实例失效自动迁移。
- Chat-Service 在处理 `StreamChat` 时：
  - 若无 `session_id` 则创建并返回；
  - 读取历史消息并构建上下文；
  - 选择目标模型实例（见“负载均衡与故障转移”）；
  - 转发到 Python `StreamInference` 并将生成的助手消息写回 Redis。

## 负载均衡与故障转移
- 每个模型维度维护实例列表与轮询游标（内存，周期性从 Consul 刷新）；
- 优先使用 `session_affinity` 指向的实例；不存在则按轮询选择；
- 调用失败策略：
  - 重试 N 次（如 2-3）按轮询选下一个实例，带指数退避；
  - 标记失败实例并短暂隔离（简单熔断窗口）；
  - 最终失败返回 `UNAVAILABLE` 并清理粘性映射。

## Python 推理服务
- 框架：`grpc.aio` + 自定义服务，实现 `StreamInference(model, messages, parameters)`，以流式返回 token 或增量文本；
- 模型管理：可通过环境变量选择后端（如 `OPENAI`/`vLLM`/`Transformers`），初期用占位回声或简单生成器实现；
- Consul 注册：启动时以 `service_name = f"llm-{MODEL_NAME}"` 注册，`meta` 包含 `model`；暴露健康端点；
- 多实例：通过 Compose 启多个副本并注册到 Consul；
- 输入兼容：支持 OpenAI 风格 `messages` 与系统/用户/助手角色，或按当前 proto 的结构体映射。

## 服务发现与注册
- `ServiceManager` 扩展：
  - 新增按 `model` 过滤实例的查询方法；
  - `GetAddressPort()` 提供 gRPC 连接地址；
  - 健康检查剔除不健康实例；
- Gateway/Chat-Service 使用 `ServiceManager` 获取 Python 推理实例列表。

## 失败重试与限流
- 保留现有 Redis 速率限制中间件（网关）；
- Chat-Service 侧新增并发限制（如每模型令牌桶）；
- 全链路设置超时（客户端与服务端）与重试策略；

## 可观测性
- 指标：
  - Go 服务集成 Prometheus（请求总数、错误率、时延、重试次数）；
  - Python 服务暴露 `prometheus_client` 指标；
- 追踪：
  - OpenTelemetry 在网关→聊天→推理的调用链中传递 trace；
- 日志：结构化 JSON，包含 `session_id`、`model`、`instance_id`、重试次数。

## 容器化与编排
- 为各服务提供 `Dockerfile`；
- `docker-compose.yml` 包含：Consul、Redis、Gateway、Chat-Service、Auth-Service、Python 推理服务的多个副本；
- 健康检查与依赖启动顺序；

## 测试与验证
- 单元测试：
  - 会话与历史读写（Redis）；
  - 负载均衡选择器与粘性路由；
  - 重试与故障转移；
- 集成测试：
  - 启动 Compose，构造两实例同名模型，发起若干 session，验证粘性与均衡分布；
  - 下线一个实例后请求仍能成功并迁移粘性映射；

## 关键改动清单（按文件）
- `cmd/gateway/internal/handler/chat.go:38-44` 与 `auth.go:34-36`：替换为 `grpc.Dial` 并使用裸地址；加入错误处理与超时。
- `shared/registry/service_manager.go:29-32,89-92`：新增 `GetAddressPort()`；`DiscoverService` 过滤健康实例并返回 `[]Instance`。
- `cmd/gateway/main.go:60-94`：修正 `ServiceManager` 初始化、路由与 handler 构造签名；提供不同端口。
- `cmd/chat-service/internal/handler/chat.go:34-93`：整合历史、粘性路由与失败重试；将生成消息持久化。
- `cmd/chat-service/internal/service/redis.go:51-150`：新增 `session_affinity` 读写接口。
- `cmd/auth-service/main.go:73,86-94`：拆分端口或移除重复 HTTP 服务器。
- `cmd/auth-service/internal/*`：统一数据库实现并引入正确驱动。
- Python 新增 `server.py`：实现 `StreamInference`、Consul 注册与指标。
- 新增容器文件：各服务 `Dockerfile` 与 `docker-compose.yml`。

## 交付阶段划分
- 阶段1：基础修复（端口、gRPC 连接、入口签名、Auth DB 驱动）。
- 阶段2：Chat-Service 的粘性路由与负载均衡，重试与超时。
- 阶段3：Python 推理服务实现与 Consul 注册，多副本部署。
- 阶段4：可观测性（Prometheus/OTEL）与日志结构化。
- 阶段5：容器化与 Compose 集成测试，完善文档与启动脚本。

## 风险与缓解
- Python 推理性能与流式稳定性：先用轻量实现与背压，再接入真实模型。
- Consul 可用性：提供本地缓存与失效回退策略。
- Redis 单点：允许外置高可用或切换到托管服务。

请确认以上计划，确认后我将按阶段开始实施并提交具体改动。