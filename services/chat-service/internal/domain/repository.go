package domain

import "context"

// ChatRepository 定义数据访问接口
// 不关心具体实现是redis，mq，还是db
type ChatRepository interface {
	SaveMessage(ctx context.Context, msg *Message) error
	SaveSession(ctx context.Context, session *Session) error
	GetSession(ctx context.Context, sessionID string) (*Session, error)
	GetSessionMessages(ctx context.Context, sessionID string, limit, offset int) ([]*Message, error)
	GetSessions(ctx context.Context, userID string, limit, offset int) ([]*Session, error)
	DeleteMessage(ctx context.Context, messageID string) error
	DeleteSession(ctx context.Context, sessionID string) error
}

// type MessageRepository interface {
// 	Save(ctx context.Context, msg *Message) error
// 	FindByID(ctx context.Context, id string) (*Message, error)
// 	FindBySessionID(ctx context.Context, sessionID string, limit, offset int) ([]*Message, error)
// 	FindByUserID(ctx context.Context, userID string, limit, offset int) ([]*Message, error)
// 	DeleteByID(ctx context.Context, id string) error
// 	DeleteBySessionID(ctx context.Context, sessionID string) error
// }

// type SessionRepository interface {
// 	Save(ctx context.Context, session *Session) error
// 	FindByID(ctx context.Context, sessionID string) (*Session, error)
// 	FindByUserID(ctx context.Context, userID string, limit, offset int) ([]*Session, error)
// 	DeleteByID(ctx context.Context, sessionID string) error
// }
