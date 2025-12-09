package adapter

import (
	"context"
	"fmt"
	"free-chat/services/chat-service/internal/domain"
	"free-chat/services/chat-service/internal/infrastructure/mq"
	"free-chat/services/chat-service/internal/infrastructure/persistence/cache"
	"free-chat/services/chat-service/internal/infrastructure/persistence/repository"
	"log"
)

type ChatRepositoryAdapter struct {
	cache       *cache.RedisCache
	msgRepo     *repository.MessageRepository
	sessionRepo *repository.SessionRepository
	producer    *mq.Producer
	consumer    *mq.Consumer
}

func NewChatRepositoryAdapter(
	cache *cache.RedisCache,
	msgRepo *repository.MessageRepository,
	sessionRepo *repository.SessionRepository,
	producer *mq.Producer,
	consumer *mq.Consumer,
) *ChatRepositoryAdapter {
	return &ChatRepositoryAdapter{
		cache:       cache,
		msgRepo:     msgRepo,
		sessionRepo: sessionRepo,
		producer:    producer,
		consumer:    consumer,
	}
}

func (adp *ChatRepositoryAdapter) SaveMessage(ctx context.Context, msg *domain.Message) error {
	if err := adp.cache.SaveMessage(ctx, msg); err != nil {
		log.Printf("[WARN] cache save message failed: %v", err)
	}
	if adp.producer != nil {
		if err := adp.producer.SendSaveMessageEvent(msg); err != nil {
			log.Printf("[ERROR] send message to MQ failed, fallback to sync write: %v", err)
			return adp.msgRepo.Save(ctx, msg)
		}
	} else {
		return adp.msgRepo.Save(ctx, msg)
	}

	return nil
}

func (adp *ChatRepositoryAdapter) SaveSession(ctx context.Context, session *domain.Session) error {
	if err := adp.cache.SaveSession(ctx, session); err != nil {
		log.Printf("[WARN] cache save session failed: %v", err)
	}
	if adp.producer != nil {
		if err := adp.producer.SendSaveSessionEvent(session); err != nil {
			log.Printf("[ERROR] send session to MQ failed, fallback to sync write: %v", err)
			return adp.sessionRepo.Save(ctx, session)
		}
	} else {
		return adp.sessionRepo.Save(ctx, session)
	}
	return nil
}

func (adp *ChatRepositoryAdapter) GetSession(ctx context.Context, sessionID string) (*domain.Session, error) {
	// 1. Try Cache
	session, err := adp.cache.GetSession(ctx, sessionID)
	if err == nil && session != nil {
		return session, nil
	}

	// 2. Try DB
	session, err = adp.sessionRepo.FindByID(ctx, sessionID)
	if err != nil {
		return nil, err
	}

	// 3. Write back to Cache
	if session != nil {
		go func(s *domain.Session) {
			_ = adp.cache.SaveSession(context.Background(), s)
		}(session)
	}

	return session, nil
}

func (adp *ChatRepositoryAdapter) GetSessionMessages(ctx context.Context, sessionID string, limit, offset int) ([]*domain.Message, error) {
	// 读缓存
	messages, err := adp.cache.GetSessionMessages(ctx, sessionID, limit, offset)
	if err == nil && len(messages) > 0 {
		return messages, nil
	}

	// Miss
	messages, err = adp.msgRepo.FindBySessionID(ctx, sessionID, limit, offset)
	if err != nil {
		return nil, err
	}

	// 回写缓存
	go func(msgs []*domain.Message) {
		for _, m := range msgs {
			_ = adp.cache.SaveMessage(context.Background(), m)
		}
	}(messages)

	return messages, nil
}

func (adp *ChatRepositoryAdapter) GetSessions(ctx context.Context, userID string, limit, offset int) ([]*domain.Session, error) {
	// 尝试从缓存读取
	sessions, err := adp.cache.GetUserSessions(ctx, userID, limit, offset)
	if err == nil && len(sessions) > 0 {
		return sessions, nil
	}

	// 缓存miss, 则从数据库读取
	sessions, err = adp.sessionRepo.FindByUserID(ctx, userID, limit, offset)
	if err != nil {
		return nil, err
	}

	// 回写缓存
	go func(ss []*domain.Session) {
		for _, s := range ss {
			_ = adp.cache.SaveSession(context.Background(), s)
		}
	}(sessions)

	return sessions, nil
}

func (adp *ChatRepositoryAdapter) DeleteMessage(ctx context.Context, messageID string) error {
	// 1. Get Message to find SessionID (needed for cache cleanup)
	msg, _ := adp.cache.GetMessage(ctx, messageID)
	if msg == nil {
		// Try DB if cache miss
		msg, _ = adp.msgRepo.FindByID(ctx, messageID)
	}

	// 2. Delete from Cache
	if msg != nil {
		if err := adp.cache.DeleteMessage(ctx, msg.SessionID, messageID); err != nil {
			log.Printf("[WARN] cache delete message failed: %v", err)
		}
	}

	// 3. Delete from DB
	return adp.msgRepo.DeleteByID(ctx, messageID)
}

func (adp *ChatRepositoryAdapter) DeleteSession(ctx context.Context, sessionID string) error {
	// 1. Get Session to find UserID
	session, _ := adp.cache.GetSession(ctx, sessionID)
	if session == nil {
		session, _ = adp.sessionRepo.FindByID(ctx, sessionID)
	}

	// 2. Delete from Cache
	if session != nil {
		if err := adp.cache.DeleteSession(ctx, session.UserID, sessionID); err != nil {
			log.Printf("[WARN] cache delete session failed: %v", err)
		}
	} else {
		// If session not found, we still try to clean up what we can guess or just rely on DB delete
		_ = adp.cache.InvalidateSessionMessages(ctx, sessionID)
	}

	// 3. Delete Messages from DB
	if err := adp.msgRepo.DeleteBySessionID(ctx, sessionID); err != nil {
		return fmt.Errorf("delete messages: %w", err)
	}

	// 4. Delete Session from DB
	return adp.sessionRepo.DeleteByID(ctx, sessionID)
}
