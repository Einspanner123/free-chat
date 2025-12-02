# Complete example: Gin + RocketMQ + gRPC (vLLM mock) integration

该工程演示一个可运行的“类 ChatGPT 后端”功能服务，重点展示 **RocketMQ**（任务队列）与 **gRPC**（模型推理服务）如何配合，并通过 **Redis Pub/Sub** 将流式 token 推回给 API（API 使用 SSE 将 token 发给浏览器客户端）。

---

## 项目结构 (单文件/目录示例)

```
chat-mvp/
├─ proto/
│  └─ vllm.proto
├─ cmd/
│  ├─ api/
│  │  └─ main.go         # Gin API: 接收请求，入队RocketMQ，提供 /stream SSE 订阅
│  ├─ dispatcher/
│  │  └─ main.go         # 消费者: 从RocketMQ取任务，调用vLLM(gRPC)，把token发布到Redis channel
│  └─ vllm_worker/
│     └─ main.go         # mock vLLM gRPC server（逐token流回）
├─ go.mod
├─ docker-compose.yml
└─ README.md
```

---

## 重要说明（简要）
- 本示例为教学与开发用途，省略了生产级别安全（TLS、认证）、持久化确认策略、复杂重试/幂等等。
- 需要本地安装 Docker 与 Docker Compose 以便快速起 RocketMQ、Redis；Go 环境用于运行服务。

---

## proto/vllm.proto

```proto
syntax = "proto3";
package vllm;

service VLLM {
  // 发送一个生成请求，服务端流式返回 token
  rpc Generate(GenerateRequest) returns (stream GenerateResponse) {}
}

message GenerateRequest {
  string request_id = 1;
  string prompt = 2;
  int32 max_tokens = 3;
}

message GenerateResponse {
  string request_id = 1;
  string token = 2;
  bool done = 3;
}
```

---

## go.mod

```
module github.com/yourorg/chat-mvp

go 1.20

require (
    github.com/apache/rocketmq-client-go/v2 v2.1.4
    github.com/gin-gonic/gin v1.9.0
    github.com/go-redis/redis/v8 v8.11.5
    github.com/google/uuid v1.3.0
    google.golang.org/grpc v1.56.0
)
```

---

## docker-compose.yml

```yaml
version: '3.8'
services:
  redis:
    image: redis:7
    ports:
      - '6379:6379'

  namesrv:
    image: apache/rocketmq:5.0.0
    command: sh -c 'bash /opt/rocketmq/bin/mqnamesrv'
    ports:
      - '9876:9876'

  broker:
    image: apache/rocketmq:5.0.0
    command: sh -c 'bash /opt/rocketmq/bin/mqbroker -n namesrv:9876'
    depends_on:
      - namesrv
    ports:
      - '10911:10911'

  # Note: go services run locally via `go run` in this demo; not containerized here

```
```

---

## cmd/api/main.go

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "time"

    "github.com/gin-gonic/gin"
    rmq "github.com/apache/rocketmq-client-go/v2"
    "github.com/apache/rocketmq-client-go/v2/producer"
    "github.com/go-redis/redis/v8"
    "github.com/google/uuid"
)

var (
    nameSrv = []string{"127.0.0.1:9876"}
    redisAddr = "127.0.0.1:6379"
)

type enqueueReq struct {
    Prompt string `json:"prompt" binding:"required"`
    MaxTok int    `json:"max_tokens"`
}

func main() {
    // RocketMQ producer
    p, err := producer.NewProducer(producer.WithNameServer(nameSrv))
    if err != nil { log.Fatalf("producer.NewProducer err=%v", err) }
    if err := p.Start(); err != nil { log.Fatalf("producer start err=%v", err) }
    defer p.Shutdown()

    // Redis client for pub/sub subscribe
    rdb := redis.NewClient(&redis.Options{Addr: redisAddr})
    ctx := context.Background()

    r := gin.Default()

    // 1) 入队：接收 prompt，写入 RocketMQ
    r.POST("/v1/chat", func(c *gin.Context) {
        var body enqueueReq
        if err := c.ShouldBindJSON(&body); err != nil {
            c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
            return
        }
        reqID := uuid.New().String()
        payload := map[string]interface{}{"request_id": reqID, "prompt": body.Prompt, "max_tokens": body.MaxTok}
        raw, _ := json.Marshal(payload)

        msg := &rmq.Message{Topic: "inference-requests", Body: raw}
        if _, err := p.SendSync(context.Background(), msg); err != nil {
            log.Printf("SendSync err=%v", err)
            c.JSON(http.StatusInternalServerError, gin.H{"error": "enqueue failed"})
            return
        }

        // return request_id; client should open SSE to /stream/:request_id
        c.JSON(http.StatusAccepted, gin.H{"request_id": reqID})
    })

    // 2) SSE endpoint: 订阅 Redis channel: response:{request_id}
    r.GET("/stream/:request_id", func(c *gin.Context) {
        requestID := c.Param("request_id")
        channel := fmt.Sprintf("response:%s", requestID)

        pubsub := rdb.Subscribe(ctx, channel)
        // ensure subscription ready
        _, err := pubsub.Receive(ctx)
        if err != nil {
            log.Printf("pubsub.Receive err=%v", err)
            c.Status(http.StatusInternalServerError)
            return
        }

        c.Writer.Header().Set("Content-Type", "text/event-stream")
        c.Writer.Header().Set("Cache-Control", "no-cache")
        c.Writer.Header().Set("Connection", "keep-alive")

        ch := pubsub.Channel()
        notify := c.Writer.CloseNotify()

        for {
            select {
            case msg := <-ch:
                // msg.Payload expected to be JSON token: {"token":"...","done":false}
                fmt.Fprintf(c.Writer, "data: %s

", msg.Payload)
                c.Writer.Flush()
                if msg.Payload == "__DONE__" {
                    pubsub.Close()
                    return
                }
            case <-notify:
                pubsub.Close()
                return
            }
        }
    })

    srv := &http.Server{Addr: ":8080", Handler: r}
    log.Println("api server listening :8080")
    if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
        log.Fatalf("ListenAndServe err=%v", err)
    }
}
```

---

## cmd/dispatcher/main.go

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "strings"
    "time"

    rmq "github.com/apache/rocketmq-client-go/v2"
    "github.com/apache/rocketmq-client-go/v2/consumer"
    "github.com/go-redis/redis/v8"
    pb "github.com/yourorg/chat-mvp/proto"
    "google.golang.org/grpc"
)

var (
    nameSrv = []string{"127.0.0.1:9876"}
    redisAddr = "127.0.0.1:6379"
)

type InferenceRequest struct {
    RequestID string `json:"request_id"`
    Prompt    string `json:"prompt"`
    MaxTok    int32  `json:"max_tokens"`
}

func main() {
    // Redis client (publish token back)
    rdb := redis.NewClient(&redis.Options{Addr: redisAddr})
    ctx := context.Background()

    // gRPC dial to vLLM mock
    conn, err := grpc.Dial("127.0.0.1:50051", grpc.WithInsecure())
    if err != nil { log.Fatalf("grpc dial err=%v", err) }
    defer conn.Close()
    client := pb.NewVLLMClient(conn)

    // RocketMQ consumer
    c, err := consumer.NewPushConsumer(
        consumer.WithNameServer(nameSrv),
        consumer.WithGroupName("dispatcher-group"),
    )
    if err != nil { log.Fatalf("NewPushConsumer err=%v", err) }

    err = c.Subscribe("inference-requests", consumer.MessageSelector{}, func(ctx context.Context, msgs ...*rmq.MessageExt) (consumer.ConsumeResult, error) {
        for _, m := range msgs {
            var req InferenceRequest
            if err := json.Unmarshal(m.Body, &req); err != nil {
                log.Printf("bad message: %v", err)
                continue
            }
            go processRequest(ctx, rdb, client, req)
        }
        return consumer.ConsumeSuccess, nil
    })
    if err := c.Start(); err != nil { log.Fatalf("consumer start err=%v", err) }
    defer c.Shutdown()

    select {}
}

func processRequest(ctx context.Context, rdb *redis.Client, client pb.VLLMClient, req InferenceRequest) {
    // call vLLM Generate (streaming)
    grpcCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
    defer cancel()

    stream, err := client.Generate(grpcCtx, &pb.GenerateRequest{RequestId: req.RequestID, Prompt: req.Prompt, MaxTokens: req.MaxTok})
    if err != nil {
        log.Printf("vllm generate err=%v", err)
        publishDone(rdb, req.RequestID)
        return
    }

    chName := fmt.Sprintf("response:%s", req.RequestID)

    for {
        resp, err := stream.Recv()
        if err != nil {
            // stream closed (io.EOF or error)
            log.Printf("stream recv err=%v", err)
            break
        }
        // package JSON payload
        payload := fmt.Sprintf(`{"token":%q,"done":%v}`, resp.Token, resp.Done)
        if err := rdb.Publish(ctx, chName, payload).Err(); err != nil {
            log.Printf("redis publish err=%v", err)
        }
        if resp.Done {
            break
        }
    }
    // signal done
    publishDone(rdb, req.RequestID)
}

func publishDone(rdb *redis.Client, requestID string) {
    ch := fmt.Sprintf("response:%s", requestID)
    // we publish a special message and then let subscribers close
    _ = rdb.Publish(context.Background(), ch, "__DONE__").Err()
}
```

---

## cmd/vllm_worker/main.go (mock server)

```go
package main

import (
    "context"
    "log"
    "net"
    "strings"
    "time"

    pb "github.com/yourorg/chat-mvp/proto"
    "google.golang.org/grpc"
)

type server struct{ pb.UnimplementedVLLMServer }

func (s *server) Generate(req *pb.GenerateRequest, stream pb.VLLM_GenerateServer) error {
    // mock: split prompt into tokens and stream them back slowly
    phrase := strings.TrimSpace(req.Prompt)
    if phrase == "" {
        phrase = "(empty prompt)"
    }
    words := strings.Split(phrase+" generated_response", " ")
    for i, w := range words {
        // simulate generation latency
        time.Sleep(80 * time.Millisecond)
        err := stream.Send(&pb.GenerateResponse{RequestId: req.RequestId, Token: w, Done: i == len(words)-1})
        if err != nil { return err }
    }
    return nil
}

func main() {
    lis, err := net.Listen("tcp", ":50051")
    if err != nil { log.Fatalf("listen err=%v", err) }
    s := grpc.NewServer()
    pb.RegisterVLLMServer(s, &server{})
    log.Println("mock vLLM listening :50051")
    if err := s.Serve(lis); err != nil { log.Fatalf("serve err=%v", err) }
}
```

---

## README.md (快速运行说明)

```
# 快速运行（开发环境）

1. 启动依赖服务（RocketMQ namesrv/broker + Redis）:

   docker compose up -d

2. 生成 gRPC 代码（在 proto 目录执行）:

   protoc --go_out=. --go-grpc_out=. proto/vllm.proto

   （确保安装了 protoc + protoc-gen-go + protoc-gen-go-grpc）

3. 启动 mock vLLM server：

   go run cmd/vllm_worker/main.go

4. 启动 dispatcher：

   go run cmd/dispatcher/main.go

5. 启动 API：

   go run cmd/api/main.go

6. 简易测试：

   curl -X POST http://localhost:8080/v1/chat -H 'Content-Type: application/json' -d '{"prompt":"hello world","max_tokens":32}'

   响应会返回 request_id，例如： {"request_id":"..."}

   然后在另一个终端用 SSE 订阅：

   curl -N http://localhost:8080/stream/<request_id>

   你将会看到逐行的 token（SSE 格式）直到 __DONE__。
```

---

## 说明与扩展建议

- **批处理**：目前 dispatcher 对每条请求都直接调用 vLLM。要提高吞吐应在 dispatcher 实现批合并（按时间窗口或请求数）并调用支持 batch 的 vLLM 接口。
- **幂等与确认**：生产环境需对 RocketMQ 消息做幂等、确认、重试策略。
- **安全**：需要对 API 做鉴权（API Key / JWT）、并对 prompt/output 做 moderation pipeline。
- **监控**：在关键位置增加 Prometheus 指标（队列长度、生成延迟、token 计数、GPU 利用率）。
- **TLS & Auth**：gRPC 和 RocketMQ 在生产网络请启用 TLS 和鉴权。

---

如果你想要，我可以：

- 将这套完整代码变成一个 `docker-compose` 可直接启动的 demo（把 Go 服务也容器化），或
- 在 dispatcher 里增加**请求批处理（batching）实现**并演示吞吐对比，或
- 把 SSE 改成 WebSocket 并加上认证。 

告诉我你想要哪个，我立刻把画布更新为那种版本（并包含可运行的 Dockerfile / compose）。
