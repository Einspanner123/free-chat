package context

import (
	"context"
	"errors"
	"strings"
	"testing"

	"free-chat/services/chat-service/internal/domain"
)

var errRemoteDown = errors.New("context-engine down")

// fakeOptimizer returns a deterministic result (or error).
type fakeOptimizer struct {
	out      string
	strategy string
	tokens   int
	err      error
}

func (f *fakeOptimizer) BuildContext(
	ctx context.Context,
	text, query, strategy string,
	budget int,
) (*domain.ContextOptimizationResult, error) {
	if f.err != nil {
		return nil, f.err
	}
	return &domain.ContextOptimizationResult{
		Context:          f.out,
		Strategy:         f.strategy,
		Tokens:           f.tokens,
		CompressionRatio: 0.7,
	}, nil
}

// recordingFallback records whether the Go-native builder was invoked.
type recordingFallback struct {
	called bool
}

func (r *recordingFallback) Build(ctx context.Context, history []*domain.Message, userMessage string, modelMaxTokens int) (*BuiltContext, error) {
	r.called = true
	return &BuiltContext{
		Messages: []*domain.Message{{Role: domain.RoleUser, Content: userMessage}},
		Strategy: "fallback",
	}, nil
}

func newRemoteBuilder(remote domain.ContextOptimizer, fallback ContextBuilder) *RemoteBuilder {
	return NewRemoteBuilder(remote, fallback, "auto")
}

func TestRemoteBuilderUsesRemoteContext(t *testing.T) {
	remote := &fakeOptimizer{out: "optimized history", strategy: "sink_topic", tokens: 100}
	fallback := &recordingFallback{}
	b := newRemoteBuilder(remote, fallback)

	built, err := b.Build(context.Background(), nil, "user question", 32768)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	if fallback.called {
		t.Error("fallback should not run when remote succeeds")
	}
	if built.Strategy != "sink_topic" {
		t.Errorf("expected strategy sink_topic, got %s", built.Strategy)
	}

	found := false
	for _, m := range built.Messages {
		if m.Role == domain.RoleSystem && m.Content == "optimized history" {
			found = true
		}
	}
	if !found {
		t.Errorf("remote context missing from messages: %+v", built.Messages)
	}
}

func TestRemoteBuilderKeepsSystemPrefixAndUserMessage(t *testing.T) {
	remote := &fakeOptimizer{out: "", strategy: "full", tokens: 0}
	b := newRemoteBuilder(remote, &recordingFallback{})

	built, err := b.Build(context.Background(), nil, "hello", 32768)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	// Position 0 = sink, position 1 = global instruction, last = user message
	if built.Messages[0].Role != domain.RoleSystem {
		t.Errorf("expected sink at position 0, got %+v", built.Messages[0])
	}
	last := built.Messages[len(built.Messages)-1]
	if last.Role != domain.RoleUser || last.Content != "hello" {
		t.Errorf("expected user message last, got %+v", last)
	}
}

func TestRemoteBuilderFallsBackOnError(t *testing.T) {
	remote := &fakeOptimizer{err: errRemoteDown}
	fallback := &recordingFallback{}
	b := newRemoteBuilder(remote, fallback)

	built, err := b.Build(context.Background(), nil, "user question", 32768)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	if !fallback.called {
		t.Error("fallback should run when remote errors")
	}
	if built.Strategy != "fallback" {
		t.Errorf("expected fallback strategy, got %s", built.Strategy)
	}
}

func TestRemoteBuilderSerializesHistoryWithRoleMarkers(t *testing.T) {
	b := newRemoteBuilder(&fakeOptimizer{out: "", strategy: "full", tokens: 0}, &recordingFallback{})
	history := []*domain.Message{
		{Role: domain.RoleUser, Content: "hi"},
		{Role: domain.RoleAssistant, Content: "hello"},
	}
	text := b.serializeHistory(history)
	if !strings.Contains(text, "user: hi") || !strings.Contains(text, "assistant: hello") {
		t.Errorf("history missing role markers: %q", text)
	}
}
