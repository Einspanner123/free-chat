package store

import (
	"fmt"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type Session struct {
	ID        uint      `json:"id" gorm:"primaryKey;autoIncrement;not null"`
	SessionID string    `json:"session_id" gorm:"uniqueIndex;size:36;not null"`
	UserID    string    `json:"user_id" gorm:"uniqueIndex;size:36;not null"`
	Title     string    `json:"title" gorm:"size:255;not null"`
	CreatedAt time.Time `json:"created_at" gorm:"autoCreateTime"`
	UpdatedAt time.Time `json:"updated_at" gorm:"autoUpdateTime"`
}

func (Session) TableName() string {
	return "sessions"
}

type SessionRepository struct {
	db *Postgres
}

func NewSessionRepository(db *Postgres) *SessionRepository {
	return &SessionRepository{db: db}
}

func (r *SessionRepository) CreateSession(userId, title string) (*Session, error) {
	session := &Session{
		UserID:    userId,
		SessionID: uuid.New().String(),
		Title:     title,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	if err := r.db.Create(&session).Error; err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	return session, nil
}

func (r *SessionRepository) GetSessionByUserID(userId string) (*Session, error) {
	var session Session
	if err := r.db.Where("user_id = ?", userId).First(&session).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, fmt.Errorf("session not found")
		}
		return nil, fmt.Errorf("failed to get session: %w", err)
	}
	return &session, nil
}
