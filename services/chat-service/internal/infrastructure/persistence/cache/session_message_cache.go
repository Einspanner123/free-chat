package cache

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"free-chat/services/chat-service/internal/domain"
	"free-chat/services/chat-service/internal/infrastructure/persistence/model"
	"time"

	"github.com/go-redis/redis/v8"
)

var ErrCacheMiss = errors.New("cache miss")

const (
	MessageTTL        = 24 * time.Hour
	SessionTTL        = 48 * time.Hour
	SessionMessageTTL = 24 * time.Hour
)

type RedisCache struct {
	client *redis.Client
}

func NewRedisCache(client *redis.Client) (*RedisCache, error) {
	err := client.Ping(context.Background()).Err()
	if err != nil {
		return nil, fmt.Errorf("ping redis: %w", err)
	}
	redisClnt := RedisCache{
		client: client,
	}
	return &redisClnt, nil
}

func (r *RedisCache) SaveMessage(ctx context.Context, message *domain.Message) error {
	msgData, err := json.Marshal(model.ToMessageModel(message))
	if err != nil {
		return err
	}
	pipe := r.client.Pipeline()
	msgKey := r.messageKey(message.ID)
	pipe.Set(ctx, msgKey, msgData, MessageTTL)
	sessionMsgKey := r.sessionMessagesKey(message.SessionID)
	pipe.ZAdd(ctx, sessionMsgKey, &redis.Z{
		Score:  float64(message.CreatedAt.UnixMicro()),
		Member: message.ID,
	})
	pipe.Expire(ctx, sessionMsgKey, SessionMessageTTL)
	_, err = pipe.Exec(ctx)
	return err
}

func (r *RedisCache) GetMessage(ctx context.Context, messageID string) (*domain.Message, error) {
	msgKey := r.messageKey(messageID)
	data, err := r.client.Get(ctx, msgKey).Result()
	if errors.Is(err, redis.Nil) {
		return nil, ErrCacheMiss
	}
	if err != nil {
		return nil, fmt.Errorf("get message from cache: %w", err)
	}

	var msgModel model.MessageModel
	if err := json.Unmarshal([]byte(data), &msgModel); err != nil {
		return nil, fmt.Errorf("unmarshal message: %w", err)
	}
	return msgModel.ToDomain(), nil
}

func (r *RedisCache) GetSessionMessages(ctx context.Context, sessionID string, limit, offset int) ([]*domain.Message, error) {
	ssMsgsKey := r.sessionMessagesKey(sessionID)
	start := int64(offset)
	stop := int64(offset + limit - 1)
	msgIDs, err := r.client.ZRevRange(ctx, ssMsgsKey, start, stop).Result()
	if err != nil {
		return nil, err
	}
	if len(msgIDs) == 0 {
		return nil, ErrCacheMiss
	}
	messages := make([]*domain.Message, 0, len(msgIDs))
	keys := make([]string, len(msgIDs))
	for i, id := range msgIDs {
		keys[i] = r.messageKey(id)
	}
	results, err := r.client.MGet(ctx, keys...).Result()
	if err != nil {
		return nil, err
	}
	for _, result := range results {
		if result == nil {
			continue
		}
		var msgModel model.MessageModel
		if err := json.Unmarshal([]byte(result.(string)), &msgModel); err != nil {
			continue
		}
		messages = append(messages, msgModel.ToDomain())
	}
	return messages, nil
}

func (r *RedisCache) SaveSession(ctx context.Context, session *domain.Session) error {
	data, err := json.Marshal(model.ToSessionModel(session))
	if err != nil {
		return fmt.Errorf("marshal session: %w", err)
	}

	pipe := r.client.Pipeline()

	sessionKey := r.sessionKey(session.ID)
	pipe.Set(ctx, sessionKey, data, SessionTTL)

	userSessionKey := r.userSessionsKey(session.UserID)
	pipe.ZAdd(ctx, userSessionKey, &redis.Z{
		Score:  float64(session.CreatedAt.UnixMicro()),
		Member: session.ID,
	})
	pipe.Expire(ctx, userSessionKey, SessionTTL)

	_, err = pipe.Exec(ctx)
	return err
}

func (r *RedisCache) GetSession(ctx context.Context, sessionID string) (*domain.Session, error) {
	sessionKey := r.sessionKey(sessionID)
	data, err := r.client.Get(ctx, sessionKey).Result()
	if errors.Is(err, redis.Nil) {
		return nil, ErrCacheMiss
	}
	if err != nil {
		return nil, fmt.Errorf("get session from cache: %w", err)
	}

	var sessionModel model.SessionModel
	if err := json.Unmarshal([]byte(data), &sessionModel); err != nil {
		return nil, fmt.Errorf("unmarshal session: %w", err)
	}
	return sessionModel.ToDomain(), nil
}

func (r *RedisCache) GetUserSessions(ctx context.Context, userID string, limit, offset int) ([]*domain.Session, error) {
	userSessionKey := r.userSessionsKey(userID)

	start := int64(offset)
	stop := int64(offset + limit - 1)

	sessionIDs, err := r.client.ZRevRange(ctx, userSessionKey, start, stop).Result()
	if err != nil {
		return nil, fmt.Errorf("get user session ids: %w", err)
	}

	if len(sessionIDs) == 0 {
		return nil, ErrCacheMiss
	}

	sessions := make([]*domain.Session, 0, len(sessionIDs))
	keys := make([]string, len(sessionIDs))
	for i, id := range sessionIDs {
		keys[i] = r.sessionKey(id)
	}

	results, err := r.client.MGet(ctx, keys...).Result()
	if err != nil {
		return nil, fmt.Errorf("mget sessions: %w", err)
	}

	for _, result := range results {
		if result == nil {
			continue
		}
		var sessionModel model.SessionModel
		if err := json.Unmarshal([]byte(result.(string)), &sessionModel); err != nil {
			continue
		}
		sessions = append(sessions, sessionModel.ToDomain())
	}

	return sessions, nil
}

func (r *RedisCache) DeleteMessage(ctx context.Context, sessionID, messageID string) error {
	pipe := r.client.Pipeline()
	pipe.Del(ctx, r.messageKey(messageID))
	pipe.ZRem(ctx, r.sessionMessagesKey(sessionID), messageID)
	_, err := pipe.Exec(ctx)
	return err
}

func (r *RedisCache) DeleteSession(ctx context.Context, userID, sessionID string) error {
	pipe := r.client.Pipeline()
	pipe.Del(ctx, r.sessionKey(sessionID))
	pipe.ZRem(ctx, r.userSessionsKey(userID), sessionID)
	pipe.Del(ctx, r.sessionMessagesKey(sessionID))
	_, err := pipe.Exec(ctx)
	return err
}

func (r *RedisCache) InvalidateSessionMessages(ctx context.Context, sessionID string) error {
	return r.client.Del(ctx, r.sessionMessagesKey(sessionID)).Err()
}

// Key generation helpers

func (r *RedisCache) messageKey(messageID string) string {
	return fmt.Sprintf("message:%s", messageID)
}

func (r *RedisCache) sessionKey(sessionID string) string {
	return fmt.Sprintf("session:%s", sessionID)
}

func (r *RedisCache) sessionMessagesKey(sessionID string) string {
	return fmt.Sprintf("session_messages:%s", sessionID)
}

func (r *RedisCache) userSessionsKey(userID string) string {
	return fmt.Sprintf("user_sessions:%s", userID)
}

func (r *RedisCache) Close() error {
	return r.client.Close()
}
