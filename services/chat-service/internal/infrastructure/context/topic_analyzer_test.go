package context

import (
	"encoding/json"
	"testing"
)

func TestTopicAnalyzerInterface(t *testing.T) {
	// 验证 TopicAnalyzer 接口定义正确：输入消息列表，输出话题列表
	var analyzer TopicAnalyzer = NewDefaultTopicAnalyzer(nil)
	if analyzer == nil {
		t.Fatal("NewDefaultTopicAnalyzer should return a non-nil analyzer")
	}
}

func TestTopicAnalyzerBuildsPrompt(t *testing.T) {
	history := []string{
		`{"role":"user","content":"什么是微服务架构？"}`,
		`{"role":"assistant","content":"微服务是一种将应用拆分为多个独立服务的架构风格。"}`,
		`{"role":"user","content":"DDD和微服务有什么关系？"}`,
		`{"role":"assistant","content":"DDD的限界上下文概念是微服务拆分的理论基础。"}`,
		`{"role":"user","content":"如何用Docker部署微服务？"}`,
	}

	analyzer := NewDefaultTopicAnalyzer(nil)
	prompt := analyzer.buildAnalysisPrompt(history)
	if prompt == "" {
		t.Fatal("buildAnalysisPrompt should return non-empty prompt")
	}

	// Prompt 中应该包含对话历史内容
	if !contains(prompt, "微服务") {
		t.Error("prompt should contain history content")
	}
	// Prompt 应该要求 JSON 输出
	if !contains(prompt, "JSON") {
		t.Error("prompt should request JSON output")
	}
}

func TestTopicAnalyzerParsesLLMResponse(t *testing.T) {
	analyzer := NewDefaultTopicAnalyzer(nil)

	llmOutput := `{
		"topics": [
			{"id": 1, "label": "微服务定义", "summary": "讨论了微服务的基本概念和特点", "msg_range": [0, 1]},
			{"id": 2, "label": "DDD关系", "summary": "解释了DDD限界上下文与微服务拆分的关系", "msg_range": [2, 3]},
			{"id": 3, "label": "Docker部署", "summary": "讨论了基于Docker的微服务部署方案", "msg_range": [4, 4]}
		]
	}`

	topics, err := analyzer.parseLLMResponse(llmOutput)
	if err != nil {
		t.Fatalf("parseLLMResponse failed: %v", err)
	}

	if len(topics) != 3 {
		t.Fatalf("expected 3 topics, got %d", len(topics))
	}

	if topics[0].Label != "微服务定义" {
		t.Errorf("expected label '微服务定义', got '%s'", topics[0].Label)
	}
	if topics[0].Summary == "" {
		t.Error("topic summary should not be empty")
	}
}

func TestTopicAnalyzerParsesInvalidResponse(t *testing.T) {
	analyzer := NewDefaultTopicAnalyzer(nil)

	// 空响应
	_, err := analyzer.parseLLMResponse("")
	if err == nil {
		t.Error("expected error for empty response")
	}

	// 非 JSON 响应
	_, err = analyzer.parseLLMResponse("I think this conversation is about...")
	if err == nil {
		t.Error("expected error for non-JSON response")
	}

	// JSON 但缺少 topics 字段
	_, err = analyzer.parseLLMResponse(`{"result": "no topics found"}`)
	if err == nil {
		t.Error("expected error for response without topics field")
	}
}

func TestTopicAnalyzerNoTopicsForShortHistory(t *testing.T) {
	analyzer := NewDefaultTopicAnalyzer(nil)

	history := []string{
		`{"role":"user","content":"你好"}`,
		`{"role":"assistant","content":"你好！有什么可以帮助你的？"}`,
	}

	// 少于 3 轮对话不应该触发分析
	if analyzer.ShouldAnalyze(history) {
		t.Error("ShouldAnalyze should return false for < 3 turns")
	}
}

func TestTopicAnalyzerTriggersAfterEnoughHistory(t *testing.T) {
	analyzer := NewDefaultTopicAnalyzer(nil)

	history := make([]string, 8)
	for i := 0; i < 8; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		history[i] = `{"role":"` + role + `","content":"message ` + string(rune('A'+i)) + `"}`
	}

	if !analyzer.ShouldAnalyze(history) {
		t.Error("ShouldAnalyze should return true for >= 6 messages (3 turns)")
	}
}

func TestTopicAnalyzerResponseWithCodeBlock(t *testing.T) {
	analyzer := NewDefaultTopicAnalyzer(nil)

	// LLM 有时会把 JSON 放在 markdown 代码块中
	llmOutput := "```json\n{\n\t\"topics\": [\n\t\t{\"id\": 1, \"label\": \"微服务\", \"summary\": \"讨论微服务架构\", \"msg_range\": [0, 2]}\n\t]\n}\n```"

	topics, err := analyzer.parseLLMResponse(llmOutput)
	if err != nil {
		t.Fatalf("parseLLMResponse failed for code-block response: %v", err)
	}

	if len(topics) != 1 {
		t.Fatalf("expected 1 topic, got %d", len(topics))
	}
	if topics[0].Label != "微服务" {
		t.Errorf("expected label '微服务', got '%s'", topics[0].Label)
	}
}

func TestSerializedTopicsAreValidJSON(t *testing.T) {
	topics := []*Topic{
		{ID: 1, Label: "微服务", Summary: "讨论了微服务架构"},
		{ID: 2, Label: "部署", Summary: "讨论了Docker部署"},
	}

	data, err := json.Marshal(topics)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var decoded []*Topic
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if len(decoded) != 2 {
		t.Errorf("expected 2 topics after round-trip, got %d", len(decoded))
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && containsStr(s, substr)
}

func containsStr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
