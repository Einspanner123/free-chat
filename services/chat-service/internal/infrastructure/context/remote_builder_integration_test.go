package context_test

import (
	"context"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"
	"time"

	"free-chat/services/chat-service/internal/domain"
	ctxpkg "free-chat/services/chat-service/internal/infrastructure/context"
	"free-chat/services/chat-service/internal/interfaces"
)

// TestRemoteBuilderCrossProcess exercises the real gRPC wire contract between the
// Go chat-service (RemoteBuilder + ContextClient) and the Python context-engine
// service running as a separate process. It proves two things the unit tests
// cannot: (1) the remote path actually builds an optimized context and returns
// routing metadata, and (2) when the Python service is down the Go-native
// builder transparently takes over (remote-primary + Go-fallback).
//
// Gated behind CTX_INTEGRATION=1: it launches the Python server, which needs the
// repo .venv and a cached Qwen3-0.6B tokenizer. This mirrors the repo convention
// of gating GPU/integration tests behind an env flag so normal `go test ./...`
// stays hermetic and fast.
func TestRemoteBuilderCrossProcess(t *testing.T) {
	if os.Getenv("CTX_INTEGRATION") != "1" {
		t.Skip("set CTX_INTEGRATION=1 to run the cross-process integration test (needs Python venv)")
	}

	repoRoot := findRepoRoot(t)
	venvPython := filepath.Join(repoRoot, ".venv", "bin", "python")
	if _, err := os.Stat(venvPython); err != nil {
		t.Skipf("python venv not found at %s: %v", venvPython, err)
	}

	port := freePort(t)
	addr := "localhost:" + port

	cmd := exec.Command(venvPython, "-m", "src.grpc_server", "--port", port)
	cmd.Dir = filepath.Join(repoRoot, "services", "context-engine")
	cmd.Env = append(os.Environ(),
		"TOKENIZER_MODEL=Qwen/Qwen3-0.6B",
		"HF_HUB_OFFLINE=1",
		"TRANSFORMERS_OFFLINE=1",
		"http_proxy=",
		"https_proxy=",
		"HTTP_PROXY=",
		"HTTPS_PROXY=",
	)
	if testing.Verbose() {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("start context-engine server: %v", err)
	}
	defer func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	}()

	waitForPort(t, addr, 60*time.Second)

	client := interfaces.NewContextClient(addr)
	defer client.Close()

	fallback := ctxpkg.NewDefaultBuilder(nil, nil)
	rb := ctxpkg.NewRemoteBuilder(client, fallback, "auto")

	history := []*domain.Message{
		{Role: domain.RoleUser, Content: "Paragraph 1: apple apple apple.\nParagraph 2: banana banana."},
	}

	// --- Happy path: remote builds and returns routing metadata. ---
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	built, err := rb.Build(ctx, history, "Which paragraph mentions apple?", 32768)
	cancel()
	if err != nil {
		t.Fatalf("remote Build failed: %v", err)
	}
	if built.Compression["remote"] != true {
		t.Errorf("happy path: expected remote=true, got %v", built.Compression["remote"])
	}
	if built.Strategy == "" {
		t.Errorf("happy path: expected non-empty strategy from remote router")
	}

	// --- Kill the Python service; Go-native builder must take over. ---
	_ = cmd.Process.Kill()
	time.Sleep(500 * time.Millisecond)

	ctx2, cancel2 := context.WithTimeout(context.Background(), 30*time.Second)
	fallbackBuilt, err := rb.Build(ctx2, history, "hello", 32768)
	cancel2()
	if err != nil {
		t.Fatalf("fallback Build should not error: %v", err)
	}
	if fallbackBuilt.Compression["remote"] == true {
		t.Errorf("fallback path: expected remote!=true, got %v", fallbackBuilt.Compression["remote"])
	}
	if fallbackBuilt.Strategy != "full" {
		t.Errorf("fallback path: expected Go-native strategy %q, got %q", "full", fallbackBuilt.Strategy)
	}
}

func findRepoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot determine test file path")
	}
	dir := filepath.Dir(file)
	for i := 0; i < 10; i++ {
		if _, err := os.Stat(filepath.Join(dir, ".venv")); err == nil {
			return dir
		}
		dir = filepath.Dir(dir)
	}
	t.Fatal("repo root (.venv) not found walking up from test file")
	return ""
}

func freePort(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("freePort: %v", err)
	}
	defer l.Close()
	return strconv.Itoa(l.Addr().(*net.TCPAddr).Port)
}

func waitForPort(t *testing.T, addr string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		c, err := net.DialTimeout("tcp", addr, 200*time.Millisecond)
		if err == nil {
			_ = c.Close()
			return
		}
		time.Sleep(200 * time.Millisecond)
	}
	t.Fatalf("context-engine server did not listen on %s within %s", addr, timeout)
}
