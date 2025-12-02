package persistence

type SessionEntity struct {
	ID string `gorm:"primaryKey;autoIncrement;column:id"`
	SessionID        string `gorm:"uniqueIndex:idx_session_id;size:36;not null;column:session_id"`
	UserID    string `gorm:"index:idx_user_id;size:36;not null;column:user_id"`
	Title     string `gorm:"type:text;not null;column:content"`
	CreatedAt time.Time `gorm:"autoCreateTime;not null;column:created_at"`
	UpdatedAt time.Time `gorm:"index;column:deleted_at"`
}

func (MessageEntity) TableName() string {
	return "sessions"
}

func (s *SessionEntity) ToDomain() *domain.Session{
	return &domain.Session{
		sessionID: s.sessionID,
		userID: s.userID,
		Title: s.Title, 
		CreatedAt: s.CreatedAt,
		UpdatedAt: s.UpdatedAt,
	}
}

type SessionRepository struct {
	db *gorm.DB
}

func (r *SessionRepository) Save(s domain.Session) error {
	session := FromDomain(s)
	if err := r.db.
		Create(session).Error; err != nil {
		return fmt.Errorf("failed to create message: %w", err)
	}
	return nil
}

func (r *SessionRepository) FindBySessionID(sessionID string, limit, offset int) ([]*domain.Session, error) {
	var entities []*SessionEntity
	if err := r.db.Where("session_id = ?", sessionID).
		Order("created_at desc").
		Limit(limit).
		Offset(offset).
		Find(&entities).Error; err != nil {
		return nil, fmt.Errorf("failed to get sessions: %w", err)
	}
	sessions := make([]*domain.Session, len(entities))
	for i, entity := range entities {
		sessions[i] = entity.ToDomain()
	}
	return sessions, nil
}