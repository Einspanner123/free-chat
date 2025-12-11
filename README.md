# Free Chat

[English](README.md) | [中文](README_CN.md)

**Free Chat** is a modern, microservices-based LLM (Large Language Model) chat application. It provides a full-stack solution for hosting and interacting with open-source LLMs, featuring a clean web interface, real-time streaming responses, and a robust backend architecture.

## ✨ Features

- **Microservices Architecture**: Built with scalability in mind, using Go for high-performance backend services and Python for LLM inference.
- **Real-time Streaming**: Experience fluid conversations with token-by-token streaming responses (Server-Sent Events).
- **Thinking Process Visualization**: Unique collapsible `<think>` tags to view or hide the LLM's internal reasoning process (Chain of Thought).
- **User Authentication**: Secure user registration and login with JWT-based sessions.
- **Chat History**: Persistent chat sessions and message history stored in PostgreSQL.
- **Service Discovery**: Automated service registration and discovery using Consul.
- **Message Queuing**: Asynchronous processing and decoupling using RocketMQ.
- **Dockerized**: Fully containerized setup for easy deployment and development.

## 🏗 Architecture

The project is organized into the following services:

| Service | Language | Description |
|---------|----------|-------------|
| **Web UI** | HTML/JS | A responsive, single-page application (SPA) frontend served by Nginx. |
| **API Gateway** | Go | The entry point for all client requests, handling routing and authentication middleware. |
| **Auth Service** | Go | Manages user registration, login, and token generation (JWT). |
| **Chat Service** | Go | Core business logic for chat sessions and message management. |
| **LLM Inference** | Python | Hosts the LLM (e.g., Qwen/Qwen3-0.6B) and exposes a gRPC interface for inference. |

**Infrastructure Components:**
- **PostgreSQL**: Primary database for user and chat data.
- **Redis**: Caching and rate limiting.
- **Consul**: Service registry and health checking.
- **RocketMQ**: Event bus for asynchronous communication.

## 🚀 Getting Started

### Prerequisites

- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed.

### Quick Start (Local Development)

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/free-chat.git
    cd free-chat
    ```

2.  **Start the application**:
    Run the following command to build and start all services:
    ```bash
    docker compose up -d --build
    ```
    *Note: The first run may take a few minutes to download the LLM model and Docker images.*

3.  **Access the application**:
    - **Web UI**: Open [http://localhost:3000](http://localhost:3000) in your browser.
    - **API Gateway**: Accessible at [http://localhost:8080](http://localhost:8080).
    - **Consul UI**: Monitor services at [http://localhost:8500](http://localhost:8500).

### Usage

1.  Open the Web UI.
2.  Register a new account (or login if you already have one).
3.  Click "**+ New Chat**" to start a conversation.
4.  Type your message and press Send.
5.  Watch the AI reply in real-time! If the model supports Chain of Thought, click "Thinking Process" to expand the reasoning steps.

## ☁️ Deployment

### Hugging Face Spaces

This project is configured for easy deployment to Hugging Face Spaces (using the Docker SDK).

See [deploy/hf-space/README.md](deploy/hf-space/README.md) for detailed instructions.

## 🛠 Configuration

The main configuration file is located at `config/config.yml`. It handles settings for:
- Database connections (Postgres, Redis)
- Service ports (gRPC, HTTP)
- LLM model selection and parameters
- JWT secrets
- RocketMQ topics

## 📂 Project Structure

```text
.
├── cmd/                # Shared command-line tools
├── config/             # Global configuration files
├── deploy/             # Deployment configurations (e.g., HF Spaces)
├── pkg/                # Shared Go packages (Proto, Utils)
├── services/           # Microservices source code
│   ├── api-gateway/    # HTTP Gateway
│   ├── auth-service/   # Authentication Service
│   ├── chat-service/   # Chat Business Logic
│   ├── llm-inference/  # Python LLM Service
│   └── web-ui/         # Frontend Static Files
└── docker-compose.yml  # Local development orchestration
```

## 📜 License

[MIT License](LICENSE)
