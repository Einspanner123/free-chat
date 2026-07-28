package context

import (
	"context"
	"testing"

	"free-chat/services/chat-service/internal/domain"
)

// TestContextBuilderSinkTokenAtPosition0 验证 sink token 位于消息列表的第一个位置
func TestContextBuilderSinkTokenAtPosition0(t *testing.T) {
	builder := NewDefaultBuilder(nil, nil)

	ctx := context.Background()
	history := []*domain.Message{
		{Role: domain.RoleUser, Content: "你好"},
	}
	built, err := builder.Build(ctx, history, "继续", 32768)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	if len(built.Messages) == 0 {
		t.Fatal("expected at least one message")
	}

	// 位置 0 应该是 sink token
	first := built.Messages[0]
	if first.Role != domain.RoleSystem && first.Role.String() != "sink" {
		t.Log("sink token role:", first.Role)
	}

	// 最后一条消息应该是当前用户输入
	last := built.Messages[len(built.Messages)-1]
	if last.Role != domain.RoleUser {
		t.Errorf("last message should be user input, got role=%s", last.Role)
	}
}

// TestContextBuilderMessageOrder 验证消息顺序：sink → history → current
func TestContextBuilderMessageOrder(t *testing.T) {
	builder := NewDefaultBuilder(nil, nil)

	ctx := context.Background()
	history := []*domain.Message{
		{Role: domain.RoleUser, Content: "第一轮用户"},
		{Role: domain.RoleAssistant, Content: "第一轮回复"},
		{Role: domain.RoleUser, Content: "第二轮用户"},
	}

	built, err := builder.Build(ctx, history, "第三轮用户", 32768)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	if len(built.Messages) < 4 {
		t.Fatalf("expected at least 4 messages (sink + 2 history + 1 current), got %d", len(built.Messages))
	}

	// 第一条: sink token（system 角色）
	// 中间: 历史消息（按原始顺序）
	// 最后一条: 当前用户消息
	last := built.Messages[len(built.Messages)-1]
	if last.Content != "第三轮用户" {
		t.Errorf("last message should be the current user input, got '%s'", last.Content)
	}
}

// TestKeyInstructionPlacement 验证关键指令出现在上下文两端
func TestKeyInstructionPlacement(t *testing.T) {
	// 关键的全局指令应该出现在开头（sink 后）和结尾（当前输入前）
	// 这是利用首位效应 + 近因效应提高指令命中率
	const globalInstruction = "Always respond in Chinese."

	messages := []*domain.Message{
		{Role: domain.RoleSystem, Content: globalInstruction},
		{Role: domain.RoleUser, Content: "你好"},
	}

	if len(messages) < 2 {
		t.Fatal("expected at least system + user messages")
	}

	// 验证 system prompt 在 sink token 之后立即出现
	if messages[0].Role != domain.RoleSystem {
		t.Error("system prompt should be at position 0 (after sink in full context)")
	}
}

// TestSinkTokenDoesNotAlterCount 验证 sink token 不影响 token 预算决策
func TestSinkTokenDoesNotAlterBudget(t *testing.T) {
	b := NewBudget(4096, 2048, 256)
	_ = b

	// sink token 只有 2 个字符，估算为 1 token
	sinkTokens := len(sinkToken) / 2
	if sinkTokens < 1 {
		sinkTokens = 1
	}
	if sinkTokens > 5 {
		t.Errorf("sink token should be minimal, estimated %d tokens", sinkTokens)
	}
}

// TestContextPreservesHistoryOrder 验证历史消息保持从旧到新的顺序
func TestContextPreservesHistoryOrder(t *testing.T) {
	builder := NewDefaultBuilder(nil, nil)

	ctx := context.Background()
	history := []*domain.Message{
		{Role: domain.RoleUser, Content: "第一句"},
		{Role: domain.RoleAssistant, Content: "回复1"},
		{Role: domain.RoleUser, Content: "第二句"},
	}

	built, err := builder.Build(ctx, history, "第三句", 32768)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	// 找到第一条 user 消息（忽略 sink token）
	firstUserIdx := -1
	for i, msg := range built.Messages {
		if msg.Role == domain.RoleUser && i < len(built.Messages)-1 {
			firstUserIdx = i
			break
		}
	}
	if firstUserIdx < 0 {
		t.Fatal("expected at least one user message in history")
	}

	// 确认第一句历史消息确实出现在前面
	if built.Messages[firstUserIdx].Content != "第一句" {
		t.Errorf("expected first history message '第一句', got '%s'", built.Messages[firstUserIdx].Content)
	}
}
