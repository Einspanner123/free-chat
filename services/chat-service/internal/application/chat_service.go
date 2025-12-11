package application

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"free-chat/services/chat-service/internal/domain"

	"github.com/google/uuid"
)

type ChatService struct {
	chatRepo     domain.ChatRepository
	modelBalance domain.ModelBalanceService
}

func NewChatService(chatRepo domain.ChatRepository, modelBalance domain.ModelBalanceService) *ChatService {
	return &ChatService{
		chatRepo:     chatRepo,
		modelBalance: modelBalance,
	}
}

// SelectBestModel 选择负载最小的模型实例并增加计数
func (s *ChatService) SelectBestModel(ctx context.Context, modelName string) (string, error) {
	return s.modelBalance.SelectAndIncreaseModelLoads(ctx, modelName)
}

// DecrementModelLoad 减少模型实例负载计数
func (s *ChatService) DecrementModelLoad(ctx context.Context, modelName, addr string) error {
	return s.modelBalance.DecrementTaskCount(ctx, modelName, addr)
}

// EnsureSession 确保会话存在
func (s *ChatService) EnsureSession(ctx context.Context, userID, sessionID, content string) (string, error) {
	if sessionID == "" {
		sessionID = uuid.New().String()
		// 创建新 Session
		session := &domain.Session{
			ID:     sessionID,
			UserID: userID,
		}
		session.SetTitle(content, 20)
		if err := s.chatRepo.SaveSession(ctx, session); err != nil {
			return "", err
		}
	}
	return sessionID, nil
}

// SaveMessage 保存消息
func (s *ChatService) SaveMessage(ctx context.Context, sessionID, userID string, role domain.Role, content string) error {
	msg := &domain.Message{
		ID:        uuid.New().String(),
		SessionID: sessionID,
		UserID:    userID,
		Role:      role,
		Content:   content,
		CreatedAt: time.Now(),
	}
	return s.chatRepo.SaveMessage(ctx, msg)
}

// GetContext 获取上下文
func (s *ChatService) GetContext(ctx context.Context, sessionID string) (string, error) {
	// 获取最近的 10 条消息
	messages, err := s.chatRepo.GetSessionMessages(ctx, sessionID, 10, 0)
	if err != nil {
		return "", fmt.Errorf("get session messages: %w", err)
	}
	if len(messages) == 0 {
		return "", nil
	}

	// 构造消息列表
	var chatMessages []map[string]string
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		chatMessages = append(chatMessages, map[string]string{
			"role":    msg.Role.String(),
			"content": msg.Content,
		})
	}

	jsonBytes, err := json.Marshal(chatMessages)
	if err != nil {
		return "", fmt.Errorf("marshal messages: %w", err)
	}

	return string(jsonBytes), nil
}

// CreateSession 创建会话
func (s *ChatService) CreateSession(ctx context.Context, userID, title string) (*domain.Session, error) {
	session := &domain.Session{
		ID:        uuid.New().String(),
		UserID:    userID,
		CreatedAt: time.Now(),
	}
	// Set title with length limit
	session.SetTitle(title, 50)

	if err := s.chatRepo.SaveSession(ctx, session); err != nil {
		return nil, err
	}
	return session, nil
}

// GetHistory 获取会话历史
func (s *ChatService) GetHistory(ctx context.Context, sessionID string, limit, offset int) ([]*domain.Message, error) {
	session, err := s.chatRepo.GetSession(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	if session == nil {
		return nil, fmt.Errorf("session not found")
	}

	return s.chatRepo.GetSessionMessages(ctx, sessionID, limit, offset)
}

// GetSessions 获取用户会话列表
func (s *ChatService) GetSessions(ctx context.Context, userID string, limit, offset int) ([]*domain.Session, error) {
	return s.chatRepo.GetSessions(ctx, userID, limit, offset)
}

// DeleteSession 删除会话
func (s *ChatService) DeleteSession(ctx context.Context, sessionID, userID string) error {
	session, err := s.chatRepo.GetSession(ctx, sessionID)
	if err != nil {
		return err
	}
	if session == nil {
		return nil // Already deleted
	}
	if session.UserID != userID {
		return fmt.Errorf("permission denied")
	}

	return s.chatRepo.DeleteSession(ctx, sessionID)
}
