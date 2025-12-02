package store

import (
	"fmt"
	"time"
)

type Summary struct {
	ID                 uint      `json:"id" gorm:"primaryKey;autoIncrement;not null"`
	UserID             string    `json:"user_id" gorm:"index;size:36;not null"`
	SessionID          string    `json:"session_id" gorm:"index;size:36;not null"`
	UserHistoryID      string    `json:"user_history_id" gorm:"uniqueIndex;size:36;not null"`
	AssistantHistoryID string    `json:"assistant_history_id" gorm:"uniqueIndex;size:36;not null"`
	SummaryContent     string    `json:"summary_content" gorm:"type:text;not null"`
	CreatedAt          time.Time `json:"created_at" gorm:"autoCreateTime"`
	UpdatedAt          time.Time `json:"updated_at" gorm:"autoUpdateTime"`
}

func (Summary) TableName() string {
	return "summaries"
}

type SummaryRepository struct {
	db *Postgres
}

func (r *SummaryRepository) CreateSummary(
	userID, sessionID, userHistoryID, assistantHistoryID, summaryContent string,
) (*Summary, error) {
	summary := &Summary{
		UserID:             userID,
		SessionID:          sessionID,
		UserHistoryID:      userHistoryID,
		AssistantHistoryID: assistantHistoryID,
		SummaryContent:     summaryContent,
	}

	// 验证唯一性约束（避免重复总结）
	var count int64
	if err := r.db.Model(&Summary{}).
		Where("user_history_id = ? OR assistant_history_id = ?",
			userHistoryID, assistantHistoryID).
		Count(&count).Error; err != nil {
		return nil, fmt.Errorf("failed to check summary uniqueness: %w", err)
	}
	if count > 0 {
		return nil, fmt.Errorf("summary already exists for this history pair")
	}

	// 插入数据库
	if err := r.db.Create(summary).Error; err != nil {
		return nil, fmt.Errorf("failed to create summary: %w", err)
	}
	return summary, nil
}

func (r *SummaryRepository) GetSummaryByUserHistoryID(historyID string) (*Summary, error) {
	var summary Summary
	if err := r.db.Where("user_history_id = ?", historyID).First(&summary).Error; err != nil {
		return nil, fmt.Errorf("summary not found: %w", err)
	}
	return &summary, nil
}

func (r *SummaryRepository) GetSummaryByAssistantHistoryID(historyID string) (*Summary, error) {
	var summary Summary
	if err := r.db.Where("assistant_history_id = ?", historyID).First(&summary).Error; err != nil {
		return nil, fmt.Errorf("summary not found: %w", err)
	}
	return &summary, nil
}

func (r *SummaryRepository) GetSummariesBySessionID(sessionID string) ([]Summary, error) {
	var summaries []Summary
	if err := r.db.Where("session_id = ?", sessionID).
		Order("created_at DESC").
		Find(&summaries).Error; err != nil {
		return nil, fmt.Errorf("failed to get summaries: %w", err)
	}
	return summaries, nil
}

func (r *SummaryRepository) GetRecentSummariesByUserID(userID string, limit, offset int) ([]Summary, error) {
	var summaries []Summary
	if err := r.db.Where("user_id = ?", userID).
		Order("created_at DESC").
		Limit(limit).
		Offset(offset).
		Find(&summaries).Error; err != nil {
		return nil, fmt.Errorf("failed to get recent summaries: %w", err)
	}
	return summaries, nil
}
