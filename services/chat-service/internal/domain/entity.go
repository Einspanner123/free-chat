package domain

import (
	"time"
)

type Role string

func (r Role) String() string {
	return string(r)
}

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
)

type Message struct {
	ID        string
	SessionID string
	UserID    string
	Role      Role
	Content   string
	CreatedAt time.Time
}

func (m *Message) IsUser() bool {
	return m.Role == RoleUser
}

type Session struct {
	ID        string
	UserID    string
	Title     string
	CreatedAt time.Time
}

func (s *Session) SetTitle(content string, maxLen int) {
	runes := []rune(content)
	if len(runes) > maxLen {
		s.Title = string(runes[:maxLen])
	} else {
		s.Title = content
	}
}

type InferenceRequest struct {
	SessionID string
	UserID    string
	Request   string
	Model     string
}

type GeneratedToken struct {
	Content string
	IsLast  bool
	Error   string
	Count   int32
}
