package repository

import (
	"context"
	"fmt"
	"free-chat/services/chat-service/internal/domain"
	"free-chat/services/chat-service/internal/infrastructure/persistence/model"

	"gorm.io/gorm"
)

type SessionRepository struct {
	db *gorm.DB
}

func NewSessionRepository(db *gorm.DB) *SessionRepository {
	return &SessionRepository{db: db}
}

func (r *SessionRepository) Save(ctx context.Context, s *domain.Session) error {
	session := model.ToSessionModel(s)
	// Create or Update
	if err := r.db.Save(session).Error; err != nil {
		return fmt.Errorf("failed to save session: %w", err)
	}
	return nil
}

func (r *SessionRepository) FindByID(ctx context.Context, sessionID string) (*domain.Session, error) {
	var sessionModel model.SessionModel
	if err := r.db.Where("session_id = ?", sessionID).First(&sessionModel).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, nil // Return nil if not found
		}
		return nil, fmt.Errorf("failed to find session: %w", err)
	}
	return sessionModel.ToDomain(), nil
}

func (r *SessionRepository) FindByUserID(ctx context.Context, userID string, limit, offset int) ([]*domain.Session, error) {
	var models []*model.SessionModel
	if err := r.db.Where("user_id = ?", userID).
		Order("created_at desc").
		Limit(limit).
		Offset(offset).
		Find(&models).Error; err != nil {
		return nil, fmt.Errorf("failed to find sessions: %w", err)
	}

	sessions := make([]*domain.Session, len(models))
	for i, m := range models {
		sessions[i] = m.ToDomain()
	}
	return sessions, nil
}

func (r *SessionRepository) DeleteByID(ctx context.Context, sessionID string) error {
	if err := r.db.Where("session_id = ?", sessionID).Delete(&model.SessionModel{}).Error; err != nil {
		return fmt.Errorf("failed to delete session: %w", err)
	}
	return nil
}
