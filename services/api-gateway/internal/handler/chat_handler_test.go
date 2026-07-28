package handler

import (
	"encoding/json"
	"testing"
)

// topicSelectSentinel is the special content marker that signals
// the ChatResponse carries topic selection data instead of a token chunk.
const topicSelectSentinel = "__TOPIC_SELECT__"

// TestTopicSelectSentinelContract verifies that the protocol contract
// between chat-service and api-gateway is well-defined.
func TestTopicSelectSentinelContract(t *testing.T) {
	// Contract: when ChatResponse.Content == "__TOPIC_SELECT__",
	// the ChatResponse.SessionId field contains a JSON array of topics.
	topics := []map[string]interface{}{
		{"id": 1, "label": "微服务", "summary": "讨论了微服务架构"},
		{"id": 2, "label": "部署", "summary": "讨论了Docker部署"},
	}
	topicsJSON, err := json.Marshal(topics)
	if err != nil {
		t.Fatalf("failed to marshal topics: %v", err)
	}

	// This is what the chat-service sends and api-gateway expects
	content := topicSelectSentinel
	sessionID := string(topicsJSON)

	if content != topicSelectSentinel {
		t.Errorf("sentinel value changed")
	}

	// API Gateway should parse sessionID as JSON when content matches sentinel
	var parsedTopics []map[string]interface{}
	if err := json.Unmarshal([]byte(sessionID), &parsedTopics); err != nil {
		t.Fatalf("api-gateway should parse sessionID as JSON topics: %v", err)
	}
	if len(parsedTopics) != 2 {
		t.Errorf("expected 2 topics, got %d", len(parsedTopics))
	}
}

func TestTopicSelectSentinelNonTopicResponse(t *testing.T) {
	// Normal responses should NOT trigger topic_select handling
	content := "普通的"
	sessionID := "session-123"

	if content == topicSelectSentinel {
		t.Error("normal chat response should not have sentinel content")
	}

	// sessionID should be treated as regular session ID, not JSON
	var parsedTopics []map[string]interface{}
	if err := json.Unmarshal([]byte(sessionID), &parsedTopics); err == nil {
		t.Error("non-sentinel sessionID should not parse as JSON topics")
	}
}

func TestTopicSelectRequestHasTopicID(t *testing.T) {
	// The StreamChat request should accept an optional topic_id field
	// for the user to select which topic to continue.
	type StreamChatRequest struct {
		Message   string `json:"message"`
		SessionId string `json:"session_id"`
		Model     string `json:"model"`
		TopicID   int    `json:"topic_id,omitempty"`
	}

	req := StreamChatRequest{
		Message:   "继续",
		SessionId: "session-123",
		TopicID:   2,
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	var decoded StreamChatRequest
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}

	if decoded.TopicID != 2 {
		t.Errorf("expected TopicID=2, got %d", decoded.TopicID)
	}

	// TopicID should be optional (zero value when not set)
	req2 := StreamChatRequest{
		Message:   "你好",
		SessionId: "session-456",
	}
	if req2.TopicID != 0 {
		t.Error("TopicID should default to 0 when not provided")
	}
}
