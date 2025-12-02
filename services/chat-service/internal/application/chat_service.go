package application

import (
	"context"
	"time"

	"free-chat/services/chat-service/internal/domain"

	"github.com/google/uuid"
)

type ChatService struct {
	repo domain.ChatRepository
	llm  domain.LLMService
}

func NewChatService(repo domain.ChatRepository, llm domain.LLMService) *ChatService {
	return &ChatService{repo: repo, llm: llm}
}

// StreamChat 核心业务逻辑
func (s *ChatService) StreamChat(ctx context.Context, userID, sessionID, content, model string) (<-chan string, <-chan error, error) {
	// 1. 确保 Session 存在
	if sessionID == "" {
		sessionID = uuid.New().String()
		// 创建新 Session
		session := &domain.Session{
			SessionID: sessionID,
			UserID:    userID,
			Title:     string([]rune(content)[:min(len(content), 20)]), // 简单取标题
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		if err := s.repo.CreateSession(ctx, session); err != nil {
			return nil, nil, err
		}
	}

	// 2. 保存用户消息 (Async: Redis + MQ)
	userMsg := &domain.Message{
		SessionID: sessionID,
		UserID:    userID,
		Role:      domain.RoleUser,
		Content:   content,
		CreatedAt: time.Now(),
	}
	if err := s.repo.SaveMessage(ctx, userMsg); err != nil {
		return nil, nil, err
	}

	// 3. 调用 LLM 服务
	tokenChan, errChan := s.llm.StreamInference(ctx, sessionID, content, model)

	// 4. 处理流式响应并聚合 (用于保存 Assistant 消息)
	// 这里我们需要返回一个 channel 给上层(Handler)，同时自己监听这个 channel 来聚合完整回复
	outTokenChan := make(chan string)
	outErrChan := make(chan error)

	go func() {
		defer close(outTokenChan)
		defer close(outErrChan)

		var fullResponse string

		for {
			select {
			case err, ok := <-errChan:
				if !ok {
					errChan = nil
					continue
				}
				outErrChan <- err
				return
			case token, ok := <-tokenChan:
				if !ok {
					// 流结束，保存 Assistant 完整消息
					asstMsg := &domain.Message{
						SessionID: sessionID,
						UserID:    userID,
						Role:      domain.RoleAssistant,
						Content:   fullResponse,
						CreatedAt: time.Now(),
					}
					s.repo.SaveMessage(context.Background(), asstMsg) // Use new context for async save
					return
				}
				fullResponse += token
				outTokenChan <- token
			case <-ctx.Done():
				return
			}
		}
	}()

	return outTokenChan, outErrChan, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
