package context

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// LLMClient 为话题分析提供 LLM 调用能力
type LLMClient interface {
	Analyze(ctx context.Context, systemPrompt, userContent string) (string, error)
}

// TopicAnalyzer identifies conversation topics by analyzing history.
type TopicAnalyzer interface {
	ShouldAnalyze(history []string) bool
	Analyze(systemPrompt, history string) ([]*Topic, error)
	buildAnalysisPrompt(history []string) string
	parseLLMResponse(response string) ([]*Topic, error)
}

type defaultTopicAnalyzer struct {
	llm LLMClient
}

func NewDefaultTopicAnalyzer(llm LLMClient) TopicAnalyzer {
	return &defaultTopicAnalyzer{llm: llm}
}

const minMessagesForAnalysis = 6

func (a *defaultTopicAnalyzer) ShouldAnalyze(history []string) bool {
	return len(history) >= minMessagesForAnalysis
}

func (a *defaultTopicAnalyzer) Analyze(systemPrompt, history string) ([]*Topic, error) {
	if a.llm == nil {
		return nil, fmt.Errorf("LLM client not available for topic analysis")
	}
	response, err := a.llm.Analyze(context.Background(), systemPrompt, history)
	if err != nil {
		return nil, fmt.Errorf("LLM analysis failed: %w", err)
	}
	return a.parseLLMResponse(response)
}

const analysisSystemPrompt = `You are a conversation analyst. Analyze the following chat history and identify distinct topics discussed.

For each topic, output a JSON object with:
- "id": a unique integer
- "label": a short 2-5 character label in Chinese
- "summary": a one-sentence summary of what was discussed
- "msg_range": [start_index, end_index] of the messages belonging to this topic

Output ONLY a JSON object with a "topics" array. No explanation, no markdown formatting.

Example:
{"topics": [{"id": 1, "label": "架构设计", "summary": "讨论了微服务的拆分原则", "msg_range": [0, 3]}]}

If there is only one topic, return {"topics": []}.`

func (a *defaultTopicAnalyzer) buildAnalysisPrompt(history []string) string {
	if len(history) == 0 {
		return ""
	}
	conversationText := strings.Join(history, "\n")
	return fmt.Sprintf("%s\n\nConversation history:\n%s\n\nTopics:", analysisSystemPrompt, conversationText)
}

func (a *defaultTopicAnalyzer) parseLLMResponse(response string) ([]*Topic, error) {
	if response == "" {
		return nil, fmt.Errorf("empty LLM response")
	}

	// Strip markdown code blocks
	cleaned := response
	if strings.HasPrefix(cleaned, "```") {
		firstNewline := strings.Index(cleaned, "\n")
		if firstNewline > 0 {
			cleaned = cleaned[firstNewline+1:]
		}
		if idx := strings.LastIndex(cleaned, "```"); idx >= 0 {
			cleaned = cleaned[:idx]
		}
		cleaned = strings.TrimSpace(cleaned)
	}

	var result struct {
		Topics []*Topic `json:"topics"`
	}
	if err := json.Unmarshal([]byte(cleaned), &result); err != nil {
		return nil, fmt.Errorf("parse JSON response: %w", err)
	}
	if result.Topics == nil {
		return nil, fmt.Errorf("response missing 'topics' field")
	}

	return result.Topics, nil
}
