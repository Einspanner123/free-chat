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
	ID        string
	SessionID string
	UserID    string
	Role      Role
	Content   string
	CreatedAt time.Time
	DeletedAt *time.Time
}

// Session 会话实体
type Session struct {
	ID        string
	UserID    string
	Title     string
	CreatedAt time.Time
	UpdatedAt time.Time
}

// Summary 总结实体
type Summary struct {
	ID                 string
	SessionID          string
	UserHistoryID      string
	AssistantHistoryID string
	Content            string
	CreatedAt          time.Time
}
