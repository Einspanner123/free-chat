package context

import (
	"context"

	"free-chat/services/chat-service/internal/domain"
)

// Budget manages token budget calculation for context window.
type Budget struct {
	MaxContextWindow int
	ReservedOutput   int
	SafetyMargin     int
	UsedTokens       int
}

func NewBudget(maxContextWindow, reservedOutput, safetyMargin int) *Budget {
	return &Budget{
		MaxContextWindow: maxContextWindow,
		ReservedOutput:   reservedOutput,
		SafetyMargin:     safetyMargin,
		UsedTokens:       0,
	}
}

func (b *Budget) Available() int {
	return b.MaxContextWindow - b.ReservedOutput - b.SafetyMargin - b.UsedTokens
}

func (b *Budget) IsExhausted() bool {
	return b.Available() <= 0
}

func (b *Budget) UsageRatio() float64 {
	total := b.MaxContextWindow - b.ReservedOutput - b.SafetyMargin
	if total <= 0 {
		return 1.0
	}
	return float64(b.UsedTokens) / float64(total)
}

// BuiltContext holds the assembled context and metadata.
type BuiltContext struct {
	Messages    []*domain.Message
	Strategy    string
	Compression map[string]interface{}
	Topics      []*Topic
	TokenBudget *Budget
}

// Topic represents a conversation topic for user selection.
type Topic struct {
	ID      string `json:"id"`
	Label   string `json:"label"`
	Summary string `json:"summary"`
}

// ContextBuilder assembles conversation context with token budget management.
// It receives the conversation history and current user message,
// applies compression if needed, and returns the assembled context.
type ContextBuilder interface {
	Build(ctx context.Context, history []*domain.Message, userMessage string, modelMaxTokens int) (*BuiltContext, error)
}

// defaultBuilder implements ContextBuilder with budget management.
type defaultBuilder struct {
	compressor Compressor
	tokenizer  TokenCounter
}

// TokenCounter provides token counting for context assembly.
type TokenCounter interface {
	Count(text string) int
}

// NewDefaultBuilder creates a ContextBuilder with compressor and tokenizer.
func NewDefaultBuilder(compressor Compressor, tokenizer TokenCounter) ContextBuilder {
	return &defaultBuilder{
		compressor: compressor,
		tokenizer:  tokenizer,
	}
}

func (b *defaultBuilder) Build(ctx context.Context, history []*domain.Message, userMessage string, modelMaxTokens int) (*BuiltContext, error) {
	_ = ctx
	_ = modelMaxTokens

	// Estimate token usage
	usedTokens := 0
	for _, msg := range history {
		if msg.TokenCount > 0 {
			usedTokens += msg.TokenCount
		} else if b.tokenizer != nil {
			usedTokens += b.tokenizer.Count(msg.Content)
		}
	}

	// Count user message
	inputTokens := 0
	if b.tokenizer != nil {
		inputTokens = b.tokenizer.Count(userMessage)
	}
	usedTokens += inputTokens

	budget := NewBudget(modelMaxTokens, 2048, 256)
	budget.UsedTokens = usedTokens

	if budget.IsExhausted() && b.compressor != nil && len(history) > 5 {
		// Compression needed
		targetBudget := budget.MaxContextWindow - budget.ReservedOutput - budget.SafetyMargin
		segments, err := b.compressor.Compress(ctx, "", history, targetBudget)
		if err == nil {
			return b.buildFromSegments(segments, userMessage, "compressed", budget)
		}
	}

	// Full context (no compression needed or compressor unavailable)
	messages := append([]*domain.Message{}, history...)
	messages = append(messages, &domain.Message{
		Role:    domain.RoleUser,
		Content: userMessage,
	})

	return &BuiltContext{
		Messages: messages,
		Strategy: "full",
		Compression: map[string]interface{}{
			"ratio":      0.0,
			"used_tokens": usedTokens,
		},
		TokenBudget: budget,
	}, nil
}

func (b *defaultBuilder) buildFromSegments(segments []*CompressedSegment, userMessage string, strategy string, budget *Budget) (*BuiltContext, error) {
	var messages []*domain.Message
	originalTokens := 0
	compressedTokens := 0

	for _, seg := range segments {
		msg := &domain.Message{
			Role:    domain.Role(seg.Role),
			Content: seg.Content,
		}
		if seg.Role != "compressed" {
			msg.Role = domain.Role(seg.Role)
		} else {
			msg.Role = domain.RoleSystem // compressed content tagged as system
		}
		messages = append(messages, msg)
		originalTokens += seg.OriginalTokens
		compressedTokens += seg.CompressedTokens
	}

	// Append current user message
	messages = append(messages, &domain.Message{
		Role:    domain.RoleUser,
		Content: userMessage,
	})

	return &BuiltContext{
		Messages: messages,
		Strategy: strategy,
		Compression: map[string]interface{}{
			"ratio":             float64(compressedTokens) / float64(max(originalTokens, 1)),
			"original_tokens":   originalTokens,
			"compressed_tokens": compressedTokens,
		},
		TokenBudget: budget,
	}, nil
}

var _ ContextBuilder = (*defaultBuilder)(nil)
