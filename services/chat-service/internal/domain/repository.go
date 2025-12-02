package domain

import "context"

// ChatRepository 定义数据访问接口
// 注意：这里不关心是写 Redis, MQ 还是 Postgres，这是 Infra 的事
type ChatRepository interface {
	// 消息相关
	SaveMessage(ctx context.Context, msg *Message) error
	GetMessages(ctx context.Context, sessionID string, limit, offset int) ([]*Message, error)

	// 会话相关
	CreateSession(ctx context.Context, session *Session) error
	GetSession(ctx context.Context, sessionID string) (*Session, error)
	GetUserSessions(ctx context.Context, userID string) ([]*Session, error)
}

// LLMService 定义大模型服务接口
type LLMService interface {
	// StreamInference 返回生成的文本通道和错误通道
	StreamInference(ctx context.Context, sessionID, message, model string) (<-chan string, <-chan error)
}
