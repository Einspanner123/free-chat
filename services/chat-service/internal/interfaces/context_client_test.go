package interfaces

import (
	"context"
	"net"
	"testing"

	contextpb "free-chat/pkg/proto/contextengine"

	"google.golang.org/grpc"
	"google.golang.org/grpc/test/bufconn"
)

// mockContextEngine implements the gRPC service for tests.
type mockContextEngine struct {
	contextpb.UnimplementedContextEngineServiceServer
}

func (m *mockContextEngine) BuildContext(
	ctx context.Context,
	req *contextpb.BuildContextRequest,
) (*contextpb.BuildContextResponse, error) {
	// Deterministic: truncate text to budget chars
	text := req.Text
	if len(text) > int(req.Budget) {
		text = text[len(text)-int(req.Budget):]
	}
	return &contextpb.BuildContextResponse{
		Context:          text,
		Strategy:         req.Strategy,
		Tokens:           int32(len(text)),
		CompressionRatio: 0.5,
	}, nil
}

func dialBufconn(t *testing.T) *grpc.ClientConn {
	t.Helper()
	lis := bufconn.Listen(1024 * 1024)
	server := grpc.NewServer()
	contextpb.RegisterContextEngineServiceServer(server, &mockContextEngine{})
	go server.Serve(lis)
	t.Cleanup(func() { server.Stop() })

	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return lis.DialContext(ctx)
		}),
		grpc.WithInsecure(),
	)
	if err != nil {
		t.Fatalf("dial failed: %v", err)
	}
	t.Cleanup(func() { conn.Close() })
	return conn
}

func TestContextClientBuildContext(t *testing.T) {
	conn := dialBufconn(t)
	// Inject the connection directly (bypass getConn which needs a real address)
	client := &ContextClient{conn: conn}

	ctx := context.Background()
	resp, err := client.BuildContext(ctx, "hello world this is a test", "", "truncation", 10)
	if err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	if len(resp.Context) > 11 { // budget 10 + margin
		t.Errorf("context too long: %d chars", len(resp.Context))
	}
	if resp.Context == "" {
		t.Error("expected non-empty context")
	}
}

func TestContextClientReturnsMetadata(t *testing.T) {
	conn := dialBufconn(t)
	client := &ContextClient{conn: conn}

	text := "hello world this is a test"
	resp, err := client.BuildContext(context.Background(), text, "q", "auto", 100)
	if err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	if resp.Strategy != "auto" {
		t.Errorf("expected strategy auto, got %q", resp.Strategy)
	}
	if resp.Tokens != len(text) {
		t.Errorf("expected tokens %d, got %d", len(text), resp.Tokens)
	}
	if resp.CompressionRatio != 0.5 {
		t.Errorf("expected compression_ratio 0.5, got %v", resp.CompressionRatio)
	}
}

func TestContextClientEmptyText(t *testing.T) {
	conn := dialBufconn(t)
	client := &ContextClient{conn: conn}

	resp, err := client.BuildContext(context.Background(), "", "", "truncation", 10)
	if err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	if resp.Context != "" {
		t.Errorf("expected empty context, got %q", resp.Context)
	}
}

func TestContextClientPreservesSuffix(t *testing.T) {
	conn := dialBufconn(t)
	client := &ContextClient{conn: conn}

	// truncation keeps last budget chars
	resp, err := client.BuildContext(context.Background(), "AAAAAAAAAABBBBBBBBBB", "", "truncation", 5)
	if err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	// mock keeps last 5 chars = "BBBBB"
	if resp.Context != "BBBBB" {
		t.Errorf("expected suffix BBBBB, got %q", resp.Context)
	}
}

func TestContextClientReusesConnection(t *testing.T) {
	conn := dialBufconn(t)
	client := &ContextClient{conn: conn}

	for i := 0; i < 3; i++ {
		_, err := client.BuildContext(context.Background(), "test", "", "truncation", 10)
		if err != nil {
			t.Fatalf("call %d failed: %v", i, err)
		}
	}
}

func TestContextClientClose(t *testing.T) {
	conn := dialBufconn(t)
	client := &ContextClient{conn: conn}
	if err := client.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
}
