package storage

import (
	"fmt"
	"free-chat/infra/database"
	"time"
)

type Session struct {
	ID        uint      `json:"id" gorm:"primaryKey;autoIncrement;not null"`
	SessionID string    `json:"session_id" gorm:"uniqueIndex;size:36;not null"`
	UserID    string    `json:"user_id" gorm:"index;size:36;not null"`
	Title     string    `json:"title" gorm:"size:255;not null"`
	CreatedAt time.Time `json:"created_at" gorm:"autoCreateTime"`
	UpdatedAt time.Time `json:"updated_at" gorm:"autoUpdateTime"`
}

func (Session) TableName() string {
	return "sessions"
}

type SessionRepository struct {
	db *database.PostgresDB
}

func NewSessionRepository(db *database.PostgresDB) *SessionRepository {
	return &SessionRepository{db: db}
}

func (r *SessionRepository) CreateSession(sessionID, userID, title string) (*Session, error) {
	session := &Session{
		SessionID: sessionID,
		UserID:    userID,
		Title:     title,
	}
	if err := r.db.Create(session).Error; err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}
	return session, nil
}

func (r *SessionRepository) GetSession(sessionID string) (*Session, error) {
	var session Session
	if err := r.db.Where("session_id = ?", sessionID).First(&session).Error; err != nil {
		return nil, fmt.Errorf("failed to get session: %w", err)
	}
	return &session, nil
}

func (r *SessionRepository) GetSessionsByUserID(userID string, limit, offset int) ([]*Session, error) {
	var sessions []*Session
	if err := r.db.Where("user_id = ?", userID).
		Order("created_at DESC").
		Limit(limit).
		Offset(offset).
		Find(&sessions).Error; err != nil {
		return nil, fmt.Errorf("failed to get sessions: %w", err)
	}
	return sessions, nil
}

func (r *SessionRepository) DeleteSession(sessionID, userID string) error {
	if err := r.db.Where("session_id = ? AND user_id = ?", sessionID, userID).Delete(&Session{}).Error; err != nil {
		return fmt.Errorf("failed to delete session: %w", err)
	}
	return nil
}

func (r *SessionRepository) UpdateSessionTitle(sessionID, userID, title string) error {
	if err := r.db.Model(&Session{}).
		Where("session_id = ? AND user_id = ?", sessionID, userID).
		Update("title", title).Error; err != nil {
		return fmt.Errorf("failed to update session title: %w", err)
	}
	return nil
}
