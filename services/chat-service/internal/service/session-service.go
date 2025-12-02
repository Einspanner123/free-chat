package service

import (
	"context"

	"free-chat/services/chat-service/internal/store"
)

type SessionService struct {
	redis    *store.RedisDB
	postgres *store.SessionRepository
}

func NewSessionService(redis *store.RedisDB, postgres *store.SessionRepository) *SessionService {
	return &SessionService{redis: redis, postgres: postgres}
}

// CreateSession 创建会话 (双写策略)
func (s *SessionService) CreateSession(ctx context.Context, userID, title string) (*store.Session, error) {
	// 1. 创建会话对象
	session, err := s.postgres.CreateSession(userID, title)
	if err != nil {
		return nil, err
	}

	// 2. 异步写入Redis (避免阻塞主流程)
	go func() {
		if s.redis != nil {
			_ = s.redis.CreateSession(context.Background(), session)
		}
	}()

	return session, nil
}

// GetUserSessions 获取用户会话 (缓存优先)
func (s *SessionService) GetUserSessions(ctx context.Context, userID string) ([]*store.Session, error) {
	// 1. 尝试从Redis获取
	if s.redis != nil {
		sessions, err := s.redis.GetUserSessions(ctx, userID)
		if err == nil && len(sessions) > 0 {
			return sessions, nil
		}
	}

	// 2. Redis失效时回源Postgres
	return s.postgres.GetSessionsByUserID(userID) // 需在SessionRepository新增此方法
}

// DeleteSession 删除会话 (双删策略)
func (s *SessionService) DeleteSession(ctx context.Context, sessionID, userID string) error {
	// 1. 删除Postgres数据
	if err := s.postgres.DeleteSession(sessionID, userID); err != nil {
		return err
	}

	// 2. 删除Redis缓存
	if s.redis != nil {
		return s.redis.DeleteSession(ctx, sessionID, userID)
	}
	return nil
}
