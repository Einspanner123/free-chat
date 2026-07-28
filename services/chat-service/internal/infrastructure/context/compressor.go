package context

import (
	"context"

	"free-chat/services/chat-service/internal/domain"
)

// CompressLevel defines the aggressiveness of compression.
type CompressLevel int

const (
	CompressLevelNone    CompressLevel = 0 // 原文保留
	CompressLevelLight   CompressLevel = 1 // 单句摘要
	CompressLevelMedium  CompressLevel = 2 // 段落摘要
	CompressLevelHeavy   CompressLevel = 3 // 标题级
	CompressLevelDiscard CompressLevel = 4 // 丢弃
)

// CompressedSegment holds a compressed message fragment.
type CompressedSegment struct {
	OriginalTokens   int           `json:"original_tokens"`
	CompressedTokens int           `json:"compressed_tokens"`
	Content          string        `json:"content"`
	Role             string        `json:"role"`
	Level            CompressLevel `json:"level"`
}

// Compressor summarizes old conversation turns to save token budget.
type Compressor interface {
	// Compress returns a compressed representation of messages.
	Compress(ctx context.Context, sessionID string, messages []*domain.Message, targetBudget int) ([]*CompressedSegment, error)
}

type defaultCompressor struct{}

// NewDefaultCompressor creates a compressor that uses heuristic truncation.
// For production, swap in a Compressor that calls Python's LLM for summarization.
func NewDefaultCompressor() Compressor {
	return &defaultCompressor{}
}

// Compress implements a simple level-based strategy:
//   - Keep last 5 messages verbatim (Level 0)
//   - Messages 6-20 get title-only (Level 3)
//   - Messages 21+ get discarded (Level 4)
func (c *defaultCompressor) Compress(ctx context.Context, sessionID string, messages []*domain.Message, targetBudget int) ([]*CompressedSegment, error) {
	_ = ctx
	_ = sessionID

	var segments []*CompressedSegment
	var totalTokens int

	for i := 0; i < len(messages); i++ {
		idx := len(messages) - 1 - i // 从最新到最旧
		m := messages[idx]
		level := c.decideLevel(i, len(messages))

		var seg *CompressedSegment
		switch level {
		case CompressLevelNone:
			seg = &CompressedSegment{
				OriginalTokens:   m.TokenCount,
				CompressedTokens: m.TokenCount,
				Content:          m.Content,
				Role:             m.Role.String(),
				Level:            level,
			}
		case CompressLevelLight, CompressLevelMedium:
			content := m.Content
			clippedLen := 100
			if level == CompressLevelMedium {
				clippedLen = 50
			}
			if len(content) > clippedLen {
				content = content[:clippedLen] + "..."
			}
			seg = &CompressedSegment{
				OriginalTokens:   m.TokenCount,
				CompressedTokens: max(len(content)/2, 1),
				Content:          content,
				Role:             m.Role.String(),
				Level:            level,
			}
		case CompressLevelHeavy:
			seg = &CompressedSegment{
				OriginalTokens:   m.TokenCount,
				CompressedTokens: 1,
				Content:          "[compressed]",
				Role:             m.Role.String(),
				Level:            level,
			}
		case CompressLevelDiscard:
			continue
		}

		segments = append(segments, seg)
		totalTokens += seg.CompressedTokens

		// 如果预算仍然不足且已经处理了足够多的旧消息，丢弃更多
		if totalTokens >= targetBudget && i >= 5 {
			break
		}
	}

	// 反转回从旧到新
	for i, j := 0, len(segments)-1; i < j; i, j = i+1, j-1 {
		segments[i], segments[j] = segments[j], segments[i]
	}

	return segments, nil
}

func (c *defaultCompressor) decideLevel(indexFromNewest, totalCount int) CompressLevel {
	if indexFromNewest < 5 {
		return CompressLevelNone
	}
	if indexFromNewest < 20 {
		return CompressLevelHeavy
	}
	return CompressLevelDiscard
}
