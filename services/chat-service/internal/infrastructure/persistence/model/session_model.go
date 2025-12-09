package model

import (
	"free-chat/services/chat-service/internal/domain"
	"time"
)

type SessionModel struct {
	ID        string    `gorm:"primaryKey;autoIncrement;column:id"`
	SessionID string    `gorm:"uniqueIndex:idx_session_id;size:36;not null;column:session_id"`
	UserID    string    `gorm:"index:idx_user_id;size:36;not null;column:user_id"`
	Title     string    `gorm:"type:text;not null;column:content"`
	CreatedAt time.Time `gorm:"autoCreateTime;not null;column:created_at"`
	UpdatedAt time.Time `gorm:"index;column:deleted_at"`
}

func (m *SessionModel) ToDomain() *domain.Session {
	return &domain.Session{
		ID:        m.ID,
		UserID:    m.UserID,
		Title:     m.Title,
		CreatedAt: m.CreatedAt,
	}
}

func ToSessionModel(d *domain.Session) *SessionModel {
	return &SessionModel{
		SessionID: d.ID,
		UserID:    d.UserID,
		Title:     d.Title,
		CreatedAt: d.CreatedAt,
	}
}
