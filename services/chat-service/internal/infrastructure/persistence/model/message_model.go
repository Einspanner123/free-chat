package model

import (
	"free-chat/services/chat-service/internal/domain"
	"time"

	"gorm.io/gorm"
)

type MessageModel struct {
	ID         uint           `gorm:"primaryKey;autoIncrement;column:id"`
	MessageID  string         `gorm:"uniqueIndex:idx_message_id;size:36;not null;column:message_id"`
	UserID     string         `gorm:"index:idx_user_id;size:36;not null;column:user_id"`
	SessionID  string         `gorm:"index:idx_session_id;size:36;not null;column:session_id"`
	Content    string         `gorm:"type:text;not null;column:content"`
	Role       string         `gorm:"size:20;not null;column:role"`
	TokenCount int            `gorm:"column:token_count;default:0"`
	CreatedAt  time.Time      `gorm:"autoCreateTime;not null;column:created_at"`
	DeletedAt  gorm.DeletedAt `gorm:"index;column:deleted_at"`
}

func (m *MessageModel) ToDomain() *domain.Message {
	return &domain.Message{
		ID:         m.MessageID,
		UserID:     m.UserID,
		SessionID:  m.SessionID,
		Role:       domain.Role(m.Role),
		Content:    m.Content,
		TokenCount: m.TokenCount,
		CreatedAt:  m.CreatedAt,
	}
}

func ToMessageModel(d *domain.Message) *MessageModel {
	return &MessageModel{
		MessageID:  d.ID,
		UserID:     d.UserID,
		SessionID:  d.SessionID,
		Content:    d.Content,
		Role:       d.Role.String(),
		TokenCount: d.TokenCount,
	}
}
