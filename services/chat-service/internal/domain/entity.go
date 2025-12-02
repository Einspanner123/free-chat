package domain

import (
	"time"
)

type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
)

// Message 核心消息实体
type Message struct {
	MessageID        string
	SessionID string
	UserID    string
	Role      Role
	Content   string
	CreatedAt time.Time
	DeletedAt *time.Time
}

// IsUser checks if the message is from a user
func (m *Message) IsUser() bool {
	return m.Role == RoleUser
}

// Session 会话实体 (Aggregate Root)
type Session struct {
	SessionID        string
	UserID    string
	Title     string
	CreatedAt time.Time
	UpdatedAt time.Time
}

// SetTitle automatically generates a title from the content if not provided
// "Good Taste": Eliminate edge cases by handling length checks internally.
func (s *Session) SetTitle(content string) {
	const maxLen = 20
	runes := []rune(content)
	if len(runes) > maxLen {
		s.Title = string(runes[:maxLen])
	} else {
		s.Title = content
	}
}
