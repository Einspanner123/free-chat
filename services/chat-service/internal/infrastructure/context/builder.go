package context

import (
	"context"
	"fmt"

	"free-chat/services/chat-service/internal/domain"
)

// sinkToken 放置在上下文头部吸收 attention sink 效应。
// 初始 token 无论内容如何都会吸收异常多的注意力，
// sink token 在位置 0 消耗这部分无效注意力，保护后续指令。
const sinkToken = "\n\n"

// globalInstruction 是每条请求重复的关键约束，
// 利用首位效应（sink 后立即出现）和近因效应（当前输入前重申）提高命中率。
const globalInstruction = "You are a helpful assistant. Respond concisely and accurately."

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
	ID      int    `json:"id"`
	Label   string `json:"label"`
	Summary string `json:"summary"`
}

// ContextBuilder assembles conversation context with token budget management.
type ContextBuilder interface {
	Build(ctx context.Context, history []*domain.Message, userMessage string, modelMaxTokens int) (*BuiltContext, error)
}

// TokenCounter provides token counting for context assembly.
type TokenCounter interface {
	Count(text string) int
}

// defaultBuilder implements ContextBuilder with budget management.
type defaultBuilder struct {
	compressor Compressor
	tokenizer  TokenCounter
}

func NewDefaultBuilder(compressor Compressor, tokenizer TokenCounter) ContextBuilder {
	return &defaultBuilder{
		compressor: compressor,
		tokenizer:  tokenizer,
	}
}

func (b *defaultBuilder) Build(ctx context.Context, history []*domain.Message, userMessage string, modelMaxTokens int) (*BuiltContext, error) {
	_ = ctx

	// Step 1: 构建注意力优化后的消息前缀
	messages := b.buildPrefixedContext(history)

	// Step 2: 估算 token 用量
	usedTokens := 0
	for _, msg := range messages {
		if msg.TokenCount > 0 {
			usedTokens += msg.TokenCount
		} else if b.tokenizer != nil {
			usedTokens += b.tokenizer.Count(msg.Content)
		}
	}
	inputTokens := 0
	if b.tokenizer != nil {
		inputTokens = b.tokenizer.Count(userMessage)
	}
	usedTokens += inputTokens

	budget := NewBudget(modelMaxTokens, 2048, 256)
	budget.UsedTokens = usedTokens

	// Step 3: 预算不足时压缩（仅压缩历史部分，保留 prefix 结构）
	if budget.IsExhausted() && b.compressor != nil && len(history) > 5 {
		targetBudget := budget.MaxContextWindow - budget.ReservedOutput - budget.SafetyMargin
		segments, err := b.compressor.Compress(ctx, "", history, targetBudget)
		if err == nil {
			return b.buildFromSegments(segments, userMessage, "compressed", budget)
		}
	}

	// Step 4: 追加当前用户消息（近因效应：关键指令在用户输入前重申）
	if len(messages) > 0 {
		// 重申关键约束（近因效应）
		messages = append(messages, &domain.Message{
			Role:    domain.RoleSystem,
			Content: globalInstruction,
		})
	}

	// 当前用户输入
	messages = append(messages, &domain.Message{
		Role:    domain.RoleUser,
		Content: userMessage,
	})

	return &BuiltContext{
		Messages: messages,
		Strategy: "full",
		Compression: map[string]interface{}{
			"ratio":       0.0,
			"used_tokens": usedTokens,
		},
		TokenBudget: budget,
	}, nil
}

// buildPrefixedContext 构建注意力优化的前缀：
//
//	位置 0: [SINK_TOKEN]       ← 吸收 attention sink
//	位置 1: [SYSTEM_PROMPT]   ← 首位效应：核心指令
//	位置 2+: [HISTORY]        ← 对话历史（从旧到新）
func (b *defaultBuilder) buildPrefixedContext(history []*domain.Message) []*domain.Message {
	var messages []*domain.Message

	// 位置 0: sink token（吸收 attention sink 效应）
	messages = append(messages, &domain.Message{
		Role:    domain.RoleSystem,
		Content: sinkToken,
	})

	// 位置 1: 全局指令（首位效应）
	messages = append(messages, &domain.Message{
		Role:    domain.RoleSystem,
		Content: fmt.Sprintf("%s\n\n%s", globalInstruction, "When in doubt, think step by step."),
	})

	// 位置 2+: 对话历史（从旧到新）
	messages = append(messages, history...)

	return messages
}

func (b *defaultBuilder) buildFromSegments(segments []*CompressedSegment, userMessage string, strategy string, budget *Budget) (*BuiltContext, error) {
	var messages []*domain.Message
	originalTokens := 0
	compressedTokens := 0

	// 保持前缀结构：sink → system → compressed history
	messages = append(messages, &domain.Message{Role: domain.RoleSystem, Content: sinkToken})
	messages = append(messages, &domain.Message{Role: domain.RoleSystem, Content: globalInstruction})

	for _, seg := range segments {
		msg := &domain.Message{
			Role:    domain.Role(seg.Role),
			Content: seg.Content,
		}
		if seg.Role == "compressed" {
			msg.Role = domain.RoleSystem
		}
		messages = append(messages, msg)
		originalTokens += seg.OriginalTokens
		compressedTokens += seg.CompressedTokens
	}

	// 近因效应：重申关键指令
	messages = append(messages, &domain.Message{Role: domain.RoleSystem, Content: globalInstruction})

	// 当前用户输入
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
