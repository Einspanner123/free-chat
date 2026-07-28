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

// NewBudget creates a budget for the given model context window.
func NewBudget(maxContextWindow, reservedOutput, safetyMargin int) *Budget {
	return &Budget{
		MaxContextWindow: maxContextWindow,
		ReservedOutput:   reservedOutput,
		SafetyMargin:     safetyMargin,
		UsedTokens:       0,
	}
}

// Available returns remaining tokens for context.
func (b *Budget) Available() int {
	return b.MaxContextWindow - b.ReservedOutput - b.SafetyMargin - b.UsedTokens
}

// IsExhausted returns true when available tokens are depleted.
func (b *Budget) IsExhausted() bool {
	return b.Available() <= 0
}

// UsageRatio returns the proportion of used tokens (0.0 to 1.0+).
func (b *Budget) UsageRatio() float64 {
	total := b.MaxContextWindow - b.ReservedOutput - b.SafetyMargin
	if total <= 0 {
		return 1.0
	}
	return float64(b.UsedTokens) / float64(total)
}

func DomainMessagesToJSON(messages []*domain.Message) (string, error) {
	// 序列化消息列表为 JSON 格式（由 chat_service.go GetContext 实现）
	// 这里只负责类型定义，实际序列化逻辑在 application 层
	return "", nil
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
type ContextBuilder interface {
	Build(ctx context.Context, sessionID, userID, userMessage string) (*BuiltContext, error)
}

// defaultBuilder is a basic implementation.
type defaultBuilder struct{}

// NewDefaultBuilder creates a ContextBuilder.
func NewDefaultBuilder() ContextBuilder {
	return &defaultBuilder{}
}

func (b *defaultBuilder) Build(ctx context.Context, sessionID, userID, userMessage string) (*BuiltContext, error) {
	_ = ctx
	_ = sessionID
	_ = userID

	messages := []*domain.Message{
		{Role: domain.RoleSystem, Content: "\n\n"},
		{Role: domain.RoleUser, Content: userMessage},
	}

	return &BuiltContext{
		Messages:    messages,
		Strategy:    "full",
		Compression: map[string]interface{}{"ratio": 0.0},
	}, nil
}

var _ ContextBuilder = (*defaultBuilder)(nil)
