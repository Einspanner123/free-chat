package context

import (
	"context"
	"log"
	"strings"

	"free-chat/services/chat-service/internal/domain"
)

// RemoteBuilder builds context via the remote context-engine service (Python),
// falling back to the Go-native builder when the remote is unavailable.
//
// The remote engine runs intent routing + retrieval/compression/layout; the
// resulting optimized context becomes the system message, preserving the
// attention-sink prefix at position 0.
type RemoteBuilder struct {
	remote   domain.ContextOptimizer
	fallback ContextBuilder
	strategy string
}

func NewRemoteBuilder(remote domain.ContextOptimizer, fallback ContextBuilder, strategy string) *RemoteBuilder {
	return &RemoteBuilder{remote: remote, fallback: fallback, strategy: strategy}
}

func (b *RemoteBuilder) Build(ctx context.Context, history []*domain.Message, userMessage string, modelMaxTokens int) (*BuiltContext, error) {
	text := b.serializeHistory(history)
	// Remote sees the same effective context budget as the Go builder.
	remoteBudget := modelMaxTokens - reservedOutputTokens - safetyMarginTokens

	result, err := b.remote.BuildContext(ctx, text, userMessage, b.strategy, remoteBudget)
	if err != nil {
		log.Printf("[WARN] context-engine unavailable, falling back to Go builder: %v", err)
		return b.fallback.Build(ctx, history, userMessage, modelMaxTokens)
	}

	messages := []*domain.Message{
		{Role: domain.RoleSystem, Content: sinkToken},
		{Role: domain.RoleSystem, Content: globalInstruction},
	}
	if result.Context != "" {
		messages = append(messages, &domain.Message{Role: domain.RoleSystem, Content: result.Context})
	}
	messages = append(messages, &domain.Message{Role: domain.RoleUser, Content: userMessage})

	return &BuiltContext{
		Messages: messages,
		Strategy: result.Strategy,
		Compression: map[string]interface{}{
			"ratio":    result.CompressionRatio,
			"tokens":   result.Tokens,
			"strategy": result.Strategy,
			"remote":   true,
		},
		TokenBudget: NewBudget(modelMaxTokens, reservedOutputTokens, safetyMarginTokens),
	}, nil
}

// serializeHistory flattens history into "role: content" lines so the remote
// intent router can detect multi-turn conversation from role markers.
func (b *RemoteBuilder) serializeHistory(history []*domain.Message) string {
	var sb strings.Builder
	for _, msg := range history {
		role := msg.Role.String()
		if role == "" {
			role = "user"
		}
		sb.WriteString(role)
		sb.WriteString(": ")
		sb.WriteString(msg.Content)
		sb.WriteString("\n")
	}
	return sb.String()
}

var _ ContextBuilder = (*RemoteBuilder)(nil)
