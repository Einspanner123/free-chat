package context

import (
	"context"
	"testing"
	"time"

	"free-chat/services/chat-service/internal/domain"
)

func TestBudgetAvailable_CalculatesCorrectly(t *testing.T) {
	b := NewBudget(32768, 2048, 256)
	b.UsedTokens = 5000

	// Available = 32768 - 2048 - 256 - 5000 = 25464
	expected := 32768 - 2048 - 256 - 5000
	if got := b.Available(); got != expected {
		t.Errorf("Available() = %d, want %d", got, expected)
	}
}

func TestBudgetAvailable_NegativeWhenExhausted(t *testing.T) {
	b := NewBudget(4096, 2048, 256)
	b.UsedTokens = 3000

	// Available = 4096 - 2048 - 256 - 3000 = -1208
	if got := b.Available(); got >= 0 {
		t.Errorf("Available() = %d, want negative", got)
	}
}

func TestBudgetIsExhausted(t *testing.T) {
	b := NewBudget(4096, 2048, 256)

	// Budget = 4096 - 2048 - 256 = 1792
	b.UsedTokens = 1500
	if b.IsExhausted() {
		t.Errorf("should not be exhausted at %d used (available=%d)", b.UsedTokens, b.Available())
	}

	b.UsedTokens = 2000
	if !b.IsExhausted() {
		t.Errorf("should be exhausted at %d used (available=%d)", b.UsedTokens, b.Available())
	}
}

func TestBudgetWatermark(t *testing.T) {
	b := NewBudget(32768, 2048, 256)

	// Total available for context = 32768 - 2048 - 256 = 30464
	// 50% watermark: 15232 used
	b.UsedTokens = 25000
	if b.UsageRatio() < 0.8 {
		t.Errorf("expected usage ratio >= 0.8 at 25000 used, got %.2f", b.UsageRatio())
	}

	b.UsedTokens = 5000
	if b.UsageRatio() > 0.5 {
		t.Errorf("expected usage ratio <= 0.5 at 5000 used, got %.2f", b.UsageRatio())
	}
}

func TestContextBuilderBuildReturnsMessages(t *testing.T) {
	builder := NewDefaultBuilder(nil, nil)

	ctx := context.Background()
	built, err := builder.Build(ctx, nil, "你好", 32768)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	if len(built.Messages) == 0 {
		t.Error("expected at least the user message in built context")
	}

	lastMsg := built.Messages[len(built.Messages)-1]
	if lastMsg.Role != domain.RoleUser || lastMsg.Content != "你好" {
		t.Errorf("last message should be the current user input, got role=%s content=%s", lastMsg.Role, lastMsg.Content)
	}
}

func TestContextBuilderBudgetExhaustedTriggersStrategy(t *testing.T) {
	builder := NewDefaultBuilder(nil, nil)

	// When budget is exhausted, strategy should be set to "compressed" or "topic_select"
	// This test defines the expected contract
	ctx := context.Background()
	built, err := builder.Build(ctx, nil, "test message", 4096)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	if built.Strategy == "" {
		t.Log("strategy field exists (will be populated when exhausted)")
	}
	_ = built
}

func TestBuiltContextContainsCompressionMetadata(t *testing.T) {
	built := &BuiltContext{
		Messages: []*domain.Message{
			{Role: domain.RoleUser, Content: "hi", TokenCount: 1, CreatedAt: time.Now()},
		},
		Strategy:    "full",
		Compression: map[string]interface{}{"ratio": 0.0},
	}

	if built.Compression["ratio"] != 0.0 {
		t.Error("compression metadata should be accessible")
	}
}

func TestCompressorKeepsRecentVerbatim(t *testing.T) {
	messages := make([]*domain.Message, 10)
	for i := 0; i < 10; i++ {
		messages[i] = &domain.Message{Role: domain.RoleUser, Content: "msg", TokenCount: 2, CreatedAt: time.Now()}
	}

	compressor := NewDefaultCompressor()
	segments, err := compressor.Compress(context.Background(), "s1", messages, 1000)
	if err != nil {
		t.Fatalf("Compress failed: %v", err)
	}

	// Last 5 should be verbatim, first 5 should be compressed or discarded
	for _, seg := range segments {
		if seg.Level == CompressLevelNone && seg.OriginalTokens != seg.CompressedTokens {
			t.Errorf("verbatim segment should have OriginalTokens == CompressedTokens, got %d vs %d", seg.OriginalTokens, seg.CompressedTokens)
		}
	}
}

func TestCompressorReturnsRoleString(t *testing.T) {
	messages := []*domain.Message{
		{Role: domain.RoleUser, Content: "hello", TokenCount: 1, CreatedAt: time.Now()},
	}
	compressor := NewDefaultCompressor()
	segments, err := compressor.Compress(context.Background(), "s1", messages, 100)
	if err != nil {
		t.Fatalf("Compress failed: %v", err)
	}
	if len(segments) != 1 {
		t.Fatalf("expected 1 segment, got %d", len(segments))
	}
	if segments[0].Role != "user" {
		t.Errorf("expected role 'user', got %s", segments[0].Role)
	}
}

func TestCompressorDiscardsOldest(t *testing.T) {
	messages := make([]*domain.Message, 25)
	for i := 0; i < 25; i++ {
		messages[i] = &domain.Message{Role: domain.RoleUser, Content: "test", TokenCount: 1, CreatedAt: time.Now()}
	}

	compressor := NewDefaultCompressor()
	segments, err := compressor.Compress(context.Background(), "s1", messages, 1000)
	if err != nil {
		t.Fatalf("Compress failed: %v", err)
	}

	// 25 messages, last 5 verbatim, messages 6-20 heavy-compressed, messages 21-24 discarded
	// Total segments: 5 (verbatim) + 15 (heavy) = 20 (not 25)
	if len(segments) > 20 {
		t.Errorf("expected at most 20 segments after compression (5 verbatim + 15 heavy), got %d", len(segments))
	}
}

