package domain

import "time"

type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
)

// Message 核心消息实体
type Message struct {
	ID        uint
	SessionID string
	UserID    string
	Role      Role
	Content   string
	CreatedAt time.Time
	DeletedAt *time.Time
}

// Session 会话实体
type Session struct {
	ID        uint
	SessionID string
	UserID    string
	Title     string
	CreatedAt time.Time
	UpdatedAt time.Time
}

// Summary 总结实体
type Summary struct {
	ID                 uint
	SessionID          string
	UserHistoryID      string
	AssistantHistoryID string
	Content            string
	CreatedAt          time.Time
}
