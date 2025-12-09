package interfaces

import (
	"context"
	"fmt"
	"io"
	"log"
	"sync"
	"time"

	llmpb "free-chat/pkg/proto/llm_inference"
	"free-chat/services/chat-service/internal/domain"

	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
)

type LLMClient struct {
	mu    sync.RWMutex
	conns map[string]*grpc.ClientConn
}

func NewLLMClient() *LLMClient {
	return &LLMClient{
		conns: make(map[string]*grpc.ClientConn),
	}
}

func (c *LLMClient) getConn(target string) (*grpc.ClientConn, error) {
	c.mu.RLock()
	conn, ok := c.conns[target]
	c.mu.RUnlock()

	if ok {
		state := conn.GetState()
		if state == connectivity.Shutdown {
			// 需要建立新连接
		} else {
			return conn, nil
		}
	}
	// 锁定map， 准备建立新连接
	c.mu.Lock()
	defer c.mu.Unlock()

	// 再次确认，防止死锁
	if conn, ok = c.conns[target]; ok {
		if conn.GetState() != connectivity.Shutdown {
			return conn, nil
		}
	}

	// 配置新连接
	conn, err := grpc.NewClient(target,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithIdleTimeout(30*time.Minute),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                20 * time.Second,
			Timeout:             10 * time.Second,
			PermitWithoutStream: true,
		}))
	if err != nil {
		return nil, err
	}

	c.conns[target] = conn
	return conn, nil
}

func (c *LLMClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	for target, conn := range c.conns {
		if err := conn.Close(); err != nil {
			log.Printf("[WARN] Failed to close connection to %s: %v", target, err)
		}
	}
	c.conns = make(map[string]*grpc.ClientConn)
	return nil
}

func (c *LLMClient) GetGeneratedToken(ctx context.Context, req *domain.InferenceRequest) (<-chan *domain.GeneratedToken, error) {
	target := req.Model
	if target == "" {
		return nil, fmt.Errorf("llm model/address is empty")
	}

	// 传递 Trace ID
	ctx = metadata.AppendToOutgoingContext(ctx, "x-trace-id", req.SessionID)

	conn, err := c.getConn(target)
	if err != nil {
		return nil, fmt.Errorf("dial llm service: %w", err)
	}

	client := llmpb.NewInferencerServiceClient(conn)
	pbReq := &llmpb.InferenceRequest{
		SessionId: req.SessionID,
		Message:   req.Request,
	}

	stream, err := client.StreamInference(ctx)
	if err != nil {
		return nil, fmt.Errorf("start stream: %w", err)
	}

	if err := stream.Send(pbReq); err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}

	outCh := make(chan *domain.GeneratedToken)

	go func() {
		defer close(outCh)
		for {
			resp, err := stream.Recv()
			if err == io.EOF {
				return
			}
			if err != nil {
				log.Printf("[ERROR] Stream recv error: %v", err)
				outCh <- &domain.GeneratedToken{
					Error:  err.Error(),
					IsLast: true,
				}
				return
			}

			token := &domain.GeneratedToken{
				Content: resp.Chunk,
				IsLast:  resp.IsFinished,
				Error:   resp.Error,
				Count:   resp.GeneratedTokens,
			}
			outCh <- token

			if resp.IsFinished {
				return
			}
		}
	}()

	return outCh, nil
}
