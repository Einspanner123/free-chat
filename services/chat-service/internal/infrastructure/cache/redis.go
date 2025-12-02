package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"free-chat/services/chat-service/internal/domain"
	"time"

	"github.com/go-redis/redis/v8"
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

func (r *RedisCache) SaveMessage(ctx context.Context, message domain.Message) error {
	msgData, err := json.Marshal(message)
	if err != nil {
		return err
	}

	msgKey := fmt.Sprintf("message:%s", message.MessageID)
	if err := r.client.
		Set(ctx, msgKey, msgData, 24*time.Hour).Err(); err != nil {
		return err
	}
	sessionMsgKey := fmt.Sprintf("session_messages:%s", message.SessionID)
	return r.client.ZAdd(ctx, sessionMsgKey, &redis.Z{
		Score:  float64(message.CreatedAt.Unix()),
		Member: message.MessageID,
	}).Err()
}

func (r *RedisCache) GetSessionMessages(ctx context.Context, sessionID string) ([]*domain.Message, error) {
	sessionMessagesKey := fmt.Sprintf("session_messages:%s", sessionID)
	messageIDs, err := r.client.ZRange(ctx, sessionMessagesKey, 0, -1).Result()
	if err != nil {
		return nil, err
	}
	messages := make([]*domain.Message, 0, len(messageIDs))
	for _, messageID := range messageIDs {
		messageKey := fmt.Sprintf("message:%s", messageID)
		messageData, err := r.client.Get(ctx, messageKey).Result()
		if err != nil {
			continue // 跳过无效的消息
		}
		var message domain.Message
		if err := json.Unmarshal([]byte(messageData), &message); err != nil {
			continue
		}
		messages = append(messages, &message)
	}
	return messages, nil
}

func (r *RedisCache) Close() error {
	return r.client.Close()
}
