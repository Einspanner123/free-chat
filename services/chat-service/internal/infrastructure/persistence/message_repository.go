package persistence

import (
	"fmt"
	"free-chat/services/chat-service/internal/domain"
	"time"

	"gorm.io/gorm"
)

type MessageEntity struct {
	ID        uint       `gorm:"primaryKey;autoIncrement;column:id"`
	MessageID string     `gorm:"uniqueIndex:idx_message_id;size:36;not null;column:message_id"`
	UserID    string     `gorm:"index:idx_user_id;size:36;not null;column:user_id"`
	SessionID string     `gorm:"index:idx_session_id;size:36;not null;column:session_id"`
	Content   string     `gorm:"type:text;not null;column:content"`
	Role      string     `gorm:"size:20;not null;column:role"`
	CreatedAt time.Time  `gorm:"autoCreateTime;not null;column:created_at"`
	DeletedAt *time.Time `gorm:"index;column:deleted_at"`
}

func (MessageEntity) TableName() string {
	return "histories"
}

func (m *MessageEntity) ToDomain() *domain.Message {
	return &domain.Message{
		MessageID:        m.MessageID,
		UserID:    m.UserID,
		Role:      domain.Role(m.Role),
		Content:   m.Content,
		CreatedAt: m.CreatedAt,
		DeletedAt: m.DeletedAt,
	}
}

func FromDomain(m domain.Message) *MessageEntity {
	return &MessageEntity{
		MessageID: m.MessageID,
		UserID:    m.UserID,
		SessionID: m.SessionID,
		Content:   m.Content,
		Role:      string(m.Role),
		CreatedAt: m.CreatedAt,
		DeletedAt: m.DeletedAt,
	}
}

type MessageRepository struct {
	db *gorm.DB
}

func NewMessageRepository(db *gorm.DB) *MessageRepository {
	return &MessageRepository{db: db}
}

func (r *MessageRepository) Save(m domain.Message) error {
	message := FromDomain(m)
	if err := r.db.
		Create(message).Error; err != nil {
		return fmt.Errorf("failed to create message: %w", err)
	}
	return nil
}

func (r *MessageRepository) FindBySessionID(sessionID string, limit, offset int) ([]*domain.Message, error) {
	var entities []*MessageEntity
	if err := r.db.Where("session_id = ?", sessionID).
		Order("created_at desc").
		Limit(limit).
		Offset(offset).
		Find(&entities).Error; err != nil {
		return nil, fmt.Errorf("failed to get messages: %w", err)
	}
	messages := make([]*domain.Message, len(entities))
	for i, entity := range entities {
		messages[i] = entity.ToDomain()
	}
	return messages, nil
}

func (r *MessageRepository) DeleteBySessionID(sessionID, userID string) error {
	if err := r.db.
		Where("session_id = ? AND user_id = ?", sessionID, userID).
		Delete(&MessageEntity{}).Error; err != nil {
		return fmt.Errorf("failed to delete messages: %w", err)
	}
	return nil
}
