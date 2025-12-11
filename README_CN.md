# Free Chat

[English](README.md) | [中文](README_CN.md)

**Free Chat** 是一个现代化的、基于微服务架构的 LLM（大语言模型）聊天应用。它提供了一个全栈解决方案，用于托管和与开源 LLM 进行交互，具有整洁的 Web 界面、实时流式响应和稳健的后端架构。

## ✨ 特性

- **微服务架构**：以可扩展性为核心构建，使用 Go 语言开发高性能后端服务，Python 处理 LLM 推理。
- **实时流式响应**：通过 Server-Sent Events (SSE) 实现逐字流式响应，体验流畅的对话。
- **用户认证**：基于 JWT 的安全用户注册和登录机制。
- **聊天记录**：聊天会话和消息历史记录持久化存储在 PostgreSQL 中。
- **服务发现**：使用 Consul 实现自动化的服务注册与发现。
- **消息队列**：使用 RocketMQ 进行异步处理和解耦。
- **Docker化**：完全容器化的设置，便于部署和开发。

## 🏗 架构

本项目包含以下服务：

| 服务 | 语言 | 描述 |
|---------|----------|-------------|
| **Web UI** | HTML/JS | 由 Nginx 服务的响应式单页应用 (SPA) 前端。 |
| **API Gateway** | Go | 所有客户端请求的入口点，处理路由和认证中间件。 |
| **Auth Service** | Go | 管理用户注册、登录和令牌生成 (JWT)。 |
| **Chat Service** | Go | 处理聊天会话和消息管理的核心业务逻辑。 |
| **LLM Inference** | Python | 托管 LLM（例如 Qwen/Qwen3-0.6B）并通过 gRPC 接口提供推理服务。 |

**基础设施组件：**
- **PostgreSQL**：用于存储用户和聊天数据的主数据库。
- **Redis**：缓存和速率限制。
- **Consul**：服务注册和健康检查。
- **RocketMQ**：用于异步通信的事件总线。

## 🚀 快速开始

### 前置条件

- 已安装 [Docker](https://www.docker.com/) 和 [Docker Compose](https://docs.docker.com/compose/)。

### 快速启动（本地开发）

1.  **克隆仓库**：
    ```bash
    git clone https://github.com/yourusername/free-chat.git
    cd free-chat
    ```

2.  **启动应用**：
    运行以下命令构建并启动所有服务：
    ```bash
    docker compose up -d --build
    ```
    *注意：首次运行可能需要几分钟时间来下载 LLM 模型和 Docker 镜像。*

3.  **访问应用**：
    - **Web UI**：在浏览器中打开 [http://localhost:3000](http://localhost:3000)。
    - **API Gateway**：地址为 [http://localhost:8080](http://localhost:8080)。
    - **Consul UI**：在 [http://localhost:8500](http://localhost:8500) 监控服务状态。

### 使用方法

1.  打开 Web UI。
2.  注册新账号（如果已有账号则直接登录）。
3.  点击 "**+ 新对话**" 开始聊天。
4.  输入消息并发送。
5.  实时观看 AI 回复！如果模型支持思维链（Chain of Thought），点击 "Thinking Process" 即可展开查看推理步骤。

## ☁️ 部署

### Hugging Face Spaces

本项目已配置为支持轻松部署到 Hugging Face Spaces（使用 Docker SDK）。

详细说明请参阅 [deploy/hf-space/README.md](deploy/hf-space/README.md)。

## 🛠 配置

主要配置文件位于 `config/config.yml`。它处理以下设置：
- 数据库连接（Postgres, Redis）
- 服务端口（gRPC, HTTP）
- LLM 模型选择和参数
- JWT 密钥
- RocketMQ 主题

## 📂 项目结构

```text
.
├── cmd/                # 共享命令行工具
├── config/             # 全局配置文件
├── deploy/             # 部署配置 (例如 HF Spaces)
├── pkg/                # 共享 Go 包 (Proto, Utils)
├── services/           # 微服务源代码
│   ├── api-gateway/    # HTTP 网关
│   ├── auth-service/   # 认证服务
│   ├── chat-service/   # 聊天业务逻辑
│   ├── llm-inference/  # Python LLM 服务
│   └── web-ui/         # 前端静态文件
└── docker-compose.yml  # 本地开发编排文件
```

## 📜 许可证

[MIT License](LICENSE)
