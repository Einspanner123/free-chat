# Free Chat

基于微服务的 LLM 聊天平台，Go 后端 + Python 推理，支持分布式部署。

[English](README.md) | [中文](README_CN.md)

## 架构

标准的微服务模式，控制平面与计算平面分离。

```mermaid
graph TD
    User((用户)) -->|HTTP| WebUI[Web UI / Nginx]
    User -->|REST| Gateway[API Gateway]
    
    subgraph "控制平面 (Control Plane)"
        Gateway -->|gRPC| Auth[Auth Service]
        Gateway -->|gRPC| Chat[Chat Service]
        Auth --> DB[(PostgreSQL)]
        Chat --> DB
        Chat --> Redis[(Redis)]
        Chat --> MQ[RocketMQ]
    end
    
    subgraph "计算平面 (Compute Plane)"
        Chat -->|gRPC| LLM[LLM Inference Service]
    end
    
    Consul[Consul 服务注册] -.->|Register/Discover| Gateway
    Consul -.->|Register| Auth
    Consul -.->|Register| Chat
    Consul -.->|Register| LLM
```

## 数据流

聊天消息的请求路径，SSE 流式传输，通过 RocketMQ 异步持久化。

```mermaid
sequenceDiagram
    participant U as User
    participant G as API Gateway
    participant C as Chat Service
    participant L as LLM Service
    participant M as RocketMQ
    
    U->>G: POST /chat/message
    G->>C: gRPC SendMessage
    
    %% Async Persistence
    par 异步持久化
        C->>M: 发布 "save-message"
    and 实时推理
        C->>L: gRPC StreamInference
        
        loop Token 生成
            L->>C: 流式响应 (Token)
            C->>G: gRPC 流式响应
            G->>U: SSE 事件 (Token)
        end
    end
    
    %% Final Save
    C->>M: 发布 "save-assistant-message"
```

## 快速开始

### 1. 单节点 (开发环境)

在单台机器上使用 Docker Compose 运行全部服务。

```bash
git clone https://github.com/einspanner/free-chat.git
cd free-chat

docker compose up -d --build
```

访问地址: `http://localhost:3000`

### 2. 分布式部署 (生产环境)

将控制平面与 GPU 计算平面分离到两台机器。

**服务器 A (控制平面):** 运行 Gateway, Auth, DB, MQ, Consul。

```bash
export ADVERTISE_IP=100.100.1.1  # 服务器 A 的 Tailscale/局域网 IP
docker-compose -f docker-compose-control.yml up -d
```

**服务器 B (GPU 计算):** 运行 Chat Service, LLM Inference。

```bash
export ADVERTISE_IP=100.100.1.2  # 服务器 B 的 Tailscale/局域网 IP
export CONTROL_PLANE_IP=100.100.1.1 # 连接到服务器 A
docker-compose -f docker-compose-compute.yml up -d
```

### 3. 运行 Qwen-3B (高性能)

默认使用 0.6B 小模型。如需更好的效果，可切换至 Qwen-3B，需至少 8GB 显存。

**方法 A：环境变量 (推荐)**

```bash
export MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
```

**方法 B：Docker Compose 覆盖**

```yaml
  llm-inference:
    environment:
      - MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
```

## 技术栈

- **Go**: 高并发服务 (Gateway, Auth, Chat)
- **Python**: PyTorch/HuggingFace 推理
- **gRPC**: 低延迟服务间通信
- **RocketMQ**: 异步消息持久化
- **Consul**: 动态服务发现
- **Tailscale**: 分布式节点间的安全网状网络

## 项目结构

```text
.
├── cmd/                # 共享命令行工具
├── config/             # 全局配置文件
├── deploy/             # 部署配置 (例如 HF Spaces)
├── pkg/                # Go 共享包 (Proto, Utils)
├── services/           # 微服务源码
│   ├── api-gateway/    # HTTP 网关
│   ├── auth-service/   # 认证服务
│   ├── chat-service/   # 聊天业务逻辑
│   ├── llm-inference/  # Python LLM 服务
│   └── web-ui/         # 前端静态文件
└── docker-compose.yml  # 本地开发编排
```
