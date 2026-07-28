package domain

import (
	"testing"
	"time"
)

func TestMessageTokenCountField(t *testing.T) {
	msg := &Message{
		ID:         "msg-1",
		SessionID:  "session-1",
		UserID:     "user-1",
		Role:       RoleUser,
		Content:    "你好，请介绍一下人工智能",
		TokenCount: 12,
		CreatedAt:  time.Now(),
	}

	if msg.TokenCount != 12 {
		t.Errorf("expected TokenCount=12, got %d", msg.TokenCount)
	}
}

func TestMessageHasTokenCountDefault(t *testing.T) {
	// 新创建的消息不应因 TokenCount 导致问题
	msg := &Message{
		ID:      "msg-2",
		Content: "test",
		Role:    RoleUser,
	}

	if msg.TokenCount != 0 {
		t.Errorf("expected default TokenCount=0, got %d", msg.TokenCount)
	}
}
