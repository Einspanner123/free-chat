package models

import "time"

type Message struct {
	ID        string    `json:"id"`
	SessionID string    `json:"session_id"`
	UserID    string    `json:"user_id"`
	Content   string    `json:"content"`
	Role      string    `json:"role"` // user, assistant, system
	Timestamp time.Time `json:"timestamp"`
}
