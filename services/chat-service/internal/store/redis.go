package store

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/go-redis/redis/v8"
)

type RedisDB struct {
	client *redis.Client
}

func NewRedisService(addr string, db int) (*RedisDB, error) {
	client := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: "",
		DB:       db,
	})

	// 测试连接
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	return &RedisDB{client: client}, nil
}

// Session管理
func (r *RedisDB) CreateSession(ctx context.Context, session *Session) error {
	sessionKey := fmt.Sprintf("session:%s", session.SessionID)
	sessionData, err := json.Marshal(session)
	if err != nil {
		return fmt.Errorf("encode json from struct failed: %w", err)
	}

	// 存储会话信息
	if err := r.client.Set(ctx, sessionKey, sessionData, 24*time.Hour).Err(); err != nil {
		return fmt.Errorf("save sessionData failed: %w", err)
	}

	// 添加到用户会话列表
	userSessionsKey := fmt.Sprintf("user_sessions:%s", session.UserID)
	err = r.client.ZAdd(ctx, userSessionsKey, &redis.Z{
		Score:  float64(session.CreatedAt.Unix()),
		Member: session.ID,
	}).Err()
	if err != nil {
		return fmt.Errorf("zadd userSession to redis failed: %w", err)
	}
	return nil
}

func (r *RedisDB) GetSession(ctx context.Context, sessionID string) (*Session, error) {
	sessionKey := fmt.Sprintf("session:%s", sessionID)
	sessionData, err := r.client.Get(ctx, sessionKey).Result()
	if err != nil {
		return nil, err
	}

	var session Session
	if err := json.Unmarshal([]byte(sessionData), &session); err != nil {
		return nil, err
	}

	return &session, nil
}

func (r *RedisDB) GetUserSessions(ctx context.Context, userID string) ([]*Session, error) {
	userSessionsKey := fmt.Sprintf("user_sessions:%s", userID)
	sessionIDs, err := r.client.ZRevRange(ctx, userSessionsKey, 0, -1).Result()
	if err != nil {
		return nil, err
	}

	sessions := make([]*Session, 0, len(sessionIDs))
	for _, sessionID := range sessionIDs {
		session, err := r.GetSession(ctx, sessionID)
		if err != nil {
			continue // 跳过无效的会话
		}
		sessions = append(sessions, session)
	}

	return sessions, nil
}

// 消息管理
// func (r *RedisService) SaveMessage(ctx context.Context, message *Message) error {
// 	messageData, err := json.Marshal(message)
// 	if err != nil {
// 		return err
// 	}

// 	// 存储消息
// 	if err := r.client.Set(ctx, messageKey, messageData, 24*time.Hour).Err(); err != nil {
// 		return err
// 	}

// 	// 添加到会话消息列表
// 	sessionMessagesKey := fmt.Sprintf("session_messages:%s", message.SessionID)
// 	return r.client.ZAdd(ctx, sessionMessagesKey, &redis.Z{
// 		Score:  float64(message.Timestamp.Unix()),
// 		Member: message.ID,
// 	}).Err()
// }

// func (r *RedisService) GetSessionMessages(ctx context.Context, sessionID string) ([]*Message, error) {
// 	sessionMessagesKey := fmt.Sprintf("session_messages:%s", sessionID)
// 	messageIDs, err := r.client.ZRange(ctx, sessionMessagesKey, 0, -1).Result()
// 	if err != nil {
// 		return nil, err
// 	}

// 	messages := make([]*Message, 0, len(messageIDs))
// 	for _, messageID := range messageIDs {
// 		messageKey := fmt.Sprintf("message:%s", messageID)
// 		messageData, err := r.client.Get(ctx, messageKey).Result()
// 		if err != nil {
// 			continue // 跳过无效的消息
// 		}

// 		var message Message
// 		if err := json.Unmarshal([]byte(messageData), &message); err != nil {
// 			continue
// 		}
// 		messages = append(messages, &message)
// 	}

// 	return messages, nil
// }

// func (r *RedisService) DeleteSession(ctx context.Context, sessionID, userID string) error {
// 	// 删除会话消息
// 	sessionMessagesKey := fmt.Sprintf("session_messages:%s", sessionID)
// 	messageIDs, err := r.client.ZRange(ctx, sessionMessagesKey, 0, -1).Result()
// 	if err == nil {
// 		for _, messageID := range messageIDs {
// 			messageKey := fmt.Sprintf("message:%s", messageID)
// 			r.client.Del(ctx, messageKey)
// 		}
// 		r.client.Del(ctx, sessionMessagesKey)
// 	}

// 	// 删除会话
// 	sessionKey := fmt.Sprintf("session:%s", sessionID)
// 	r.client.Del(ctx, sessionKey)

// 	// 从用户会话列表中移除
// 	userSessionsKey := fmt.Sprintf("user_sessions:%s", userID)
// 	return r.client.ZRem(ctx, userSessionsKey, sessionID).Err()
// }

func (r *RedisDB) Close() error {
	return r.client.Close()
}
