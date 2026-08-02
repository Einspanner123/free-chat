package interfaces

import (
	"context"
	"fmt"
	"sync"
	"time"

	contextpb "free-chat/pkg/proto/contextengine"
	"free-chat/services/chat-service/internal/domain"

	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
)

// ContextClient calls the context-engine service (Python) via gRPC.
// Builds optimized contexts under a token budget before LLM inference.
type ContextClient struct {
	mu     sync.RWMutex
	conn   *grpc.ClientConn
	target string
}

// NewContextClient creates a client for the context-engine service.
func NewContextClient(target string) *ContextClient {
	return &ContextClient{target: target}
}

func (c *ContextClient) getConn() (*grpc.ClientConn, error) {
	c.mu.RLock()
	if c.conn != nil && c.conn.GetState() != connectivity.Shutdown {
		conn := c.conn
		c.mu.RUnlock()
		return conn, nil
	}
	c.mu.RUnlock()

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.conn != nil && c.conn.GetState() != connectivity.Shutdown {
		return c.conn, nil
	}

	conn, err := grpc.NewClient(c.target,
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
	c.conn = conn
	return conn, nil
}

// BuildContext calls the remote context-engine to optimize a context.
// Implements domain.ContextOptimizer.
func (c *ContextClient) BuildContext(
	ctx context.Context,
	text, query, strategy string,
	budget int,
) (string, error) {
	conn, err := c.getConn()
	if err != nil {
		return "", fmt.Errorf("connect context-engine: %w", err)
	}
	client := contextpb.NewContextEngineServiceClient(conn)
	req := &contextpb.BuildContextRequest{
		Text:     text,
		Query:    query,
		Strategy: strategy,
		Budget:   int32(budget),
	}
	resp, err := client.BuildContext(ctx, req)
	if err != nil {
		return "", fmt.Errorf("BuildContext RPC: %w", err)
	}
	return resp.Context, nil
}

func (c *ContextClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

var _ domain.ContextOptimizer = (*ContextClient)(nil)
