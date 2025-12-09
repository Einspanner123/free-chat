package repository

import (
	"context"
	"fmt"
	"free-chat/services/chat-service/internal/domain"
	"free-chat/services/chat-service/internal/infrastructure/persistence/model"

	"gorm.io/gorm"
)

type MessageRepository struct {
	db *gorm.DB
}

func NewMessageRepository(db *gorm.DB) *MessageRepository {
	return &MessageRepository{db: db}
}

func (r *MessageRepository) Save(ctx context.Context, m *domain.Message) error {
	message := model.ToMessageModel(m)
	if err := r.db.Create(message).Error; err != nil {
		return fmt.Errorf("failed to create message: %w", err)
	}
	return nil
}

func (r *MessageRepository) FindByID(ctx context.Context, id string) (*domain.Message, error) {
	var model model.MessageModel
	if err := r.db.Where("id = ?", id).First(&model).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to find message: %w", err)
	}
	return model.ToDomain(), nil
}

func (r *MessageRepository) FindBySessionID(ctx context.Context, sessionID string, limit, offset int) ([]*domain.Message, error) {
	var models []*model.MessageModel
	if err := r.db.Where("session_id = ?", sessionID).
		Order("created_at desc").
		Limit(limit).
		Offset(offset).
		Find(&models).Error; err != nil {
		return nil, fmt.Errorf("failed to get messages: %w", err)
	}
	messages := make([]*domain.Message, len(models))
	for i, entity := range models {
		messages[i] = entity.ToDomain()
	}
	return messages, nil
}

func (r *MessageRepository) FindByUserID(ctx context.Context, userID string, limit, offset int) ([]*domain.Message, error) {
	var models []*model.MessageModel
	if err := r.db.Where("user_id = ?", userID).
		Order("created_at desc").
		Limit(limit).
		Offset(offset).
		Find(&models).Error; err != nil {
		return nil, fmt.Errorf("failed to get messages: %w", err)
	}
	messages := make([]*domain.Message, len(models))
	for i, entity := range models {
		messages[i] = entity.ToDomain()
	}
	return messages, nil
}

func (r *MessageRepository) DeleteByID(ctx context.Context, id string) error {
	if err := r.db.Where("id = ?", id).Delete(&model.MessageModel{}).Error; err != nil {
		return fmt.Errorf("failed to delete message: %w", err)
	}
	return nil
}

func (r *MessageRepository) DeleteBySessionID(ctx context.Context, sessionID string) error {
	if err := r.db.
		Where("session_id = ?", sessionID).
		Delete(&model.MessageModel{}).Error; err != nil {
		return fmt.Errorf("failed to delete messages: %w", err)
	}
	return nil
}
