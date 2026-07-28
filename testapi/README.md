# FreeChat API Collection (Bruno)

This directory contains a [Bruno](https://www.usebruno.com/) API collection for testing FreeChat services.

## Prerequisites

- [Bruno](https://www.usebruno.com/downloads) desktop app (open-source, free)
- FreeChat services running locally (`docker compose up -d`)

## Usage

1. Open Bruno → **Open Collection** → select `testapi/`
2. Set variables in the collection editor:
   - `base_url`: API Gateway address (default: `http://localhost:8080`)
   - `jwt_token`: Token obtained from login (run **Login** first)
   - `refresh_token`: Token from login response
   - `session_id`: UUID from **Create Session** response
3. Execute requests in order:
   ```
   Health Check  →  Login  →  Create Session  →  Stream Chat
   ```

## Request Flow

```
health (GET /health)
  ↓
register (POST /auth/register) — one-time setup
  ↓
login (POST /auth/login) — obtain jwt_token
  ↓
create_session (POST /chat/sessions) — obtain session_id
  ↓
send_message (POST /chat/sessions/messages) — SSE streaming chat
  or
streamchat (POST /chat/sessions/stream) — SSE streaming chat
  ↓
get_sessions (GET /chat/sessions) — list sessions
get_history (GET /chat/sessions/:id/history) — session messages
delete_session (DELETE /chat/sessions/:id) — remove session
refresh (POST /auth/refresh) — refresh jwt_token
```

## Endpoints

| Method | Path | File |
|--------|------|------|
| GET | `/health` | `health.bru` |
| POST | `/api/v1/auth/login` | `auth-service/login.bru` |
| POST | `/api/v1/auth/register` | `auth-service/register.bru` |
| POST | `/api/v1/auth/refresh` | `auth-service/refresh.bru` |
| POST | `/api/v1/chat/sessions` | `chat-service/create_session.bru` |
| GET | `/api/v1/chat/sessions` | `chat-service/get_sessions.bru` |
| GET | `/api/v1/chat/sessions/:id/history` | `chat-service/get_history.bru` |
| DELETE | `/api/v1/chat/sessions/:id` | `chat-service/delete_session.bru` |
| POST | `/api/v1/chat/sessions/messages` | `chat-service/send_message.bru` |
| POST | `/api/v1/chat/sessions/stream` | `streamchat.bru` |

## Variables

Defined in `collection.bru`:

| Variable | Description | Source |
|----------|-------------|--------|
| `base_url` | API Gateway URL | Default: `http://localhost:8080` |
| `jwt_token` | JWT access token | Login response → `access_token` |
| `refresh_token` | JWT refresh token | Login response → `refresh_token` |
| `session_id` | Active session UUID | Create Session response → `session_id` |
