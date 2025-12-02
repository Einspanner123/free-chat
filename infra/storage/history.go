package storage

import (
	"fmt"
	"free-chat/infra/database"
	"time"
)

type History struct {
	ID        uint       `json:"id" gorm:"primaryKey;autoIncrement"`
	HistoryID string     `json:"history_id" gorm:"index:idx_session_user,priority:1;size:255;not null"`
	UserID    string     `json:"user_id" gorm:"index:idx_session_user,priority:2;index;size:255;not null"`
	SessionID string     `json:"session_id" gorm:"index:idx_session_user,priority:3;size:255;not null"`
	Content   string     `json:"content" gorm:"type:text;not null"`
	Role      string     `json:"role" gorm:"type:varchar(20);not null;check:role IN ('user','assistant','system')"`
	CreatedAt time.Time  `json:"created_at" gorm:"autoCreateTime"`
	DeletedAt *time.Time `json:"deleted_at,omitempty" gorm:"index"`
}

func (History) TableName() string {
	return "histories"
}

type HistoryRepository struct {
	db *database.PostgresDB
}

func NewHistoryRepository(db *database.PostgresDB) *HistoryRepository {
	return &HistoryRepository{db: db}
}

func (r *HistoryRepository) CreateHistory(userID, sessionID, content, role string) (*History, error) {
	message := &History{
		UserID:    userID,
		SessionID: sessionID,
		Content:   content,
		Role:      role,
	}
	if err := r.db.Create(message).Error; err != nil {
		return nil, fmt.Errorf("failed to create message: %w", err)
	}
	return message, nil
}

func (r *HistoryRepository) GetHistoryByUserID(userID string, limit, offset int) ([]*History, error) {
	var messages []*History
	if err := r.db.Where("user_id = ?", userID).
		Order("created_at desc").
		Limit(limit).
		Offset(offset).
		Find(&messages).Error; err != nil {
		return nil, fmt.Errorf("failed to get messages: %w", err)
	}
	return messages, nil
}

func (r *HistoryRepository) GetHistoryBySession(sessionID string, limit, offset int) ([]*History, error) {
	var messages []*History
	if err := r.db.Where("session_id = ?", sessionID).
		Order("created_at desc").
		Limit(limit).
		Offset(offset).
		Find(&messages).Error; err != nil {
		return nil, fmt.Errorf("failed to get messages: %w", err)
	}
	return messages, nil
}

func (r *HistoryRepository) DeleteBySessionID(sessionID, userID string) error {
	if err := r.db.Where("session_id = ? AND user_id = ?", sessionID, userID).Delete(&History{}).Error; err != nil {
		return fmt.Errorf("failed to delete messages: %w", err)
	}
	return nil
}
