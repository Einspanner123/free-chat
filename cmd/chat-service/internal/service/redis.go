package service

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/go-redis/redis/v8"
)

type RedisService struct {
	client *redis.Client
}

type Message struct {
	SessionID string    `json:"session_id"`
	UserID    string    `json:"user_id"`
	Content   string    `json:"content"`
	Role      string    `json:"role"` // user, assistant, system
	Timestamp time.Time `json:"timestamp"`
}

type Session struct {
	ID        string    `json:"id"`
	UserID    string    `json:"user_id"`
	Title     string    `json:"title"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

func NewRedisService(addr string, db int) (*RedisService, error) {
	client := redis.NewClient(&redis.Options{
		Addr: addr,
		// Password: password,
		DB: db,
	})

	// 测试连接
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %v", err)
	}

	return &RedisService{client: client}, nil
}

// Session管理
func (r *RedisService) CreateSession(ctx context.Context, session *Session) error {
	sessionKey := fmt.Sprintf("session:%s", session.ID)
	sessionData, err := json.Marshal(session)
	if err != nil {
		return err
	}

	// 存储会话信息
	if err := r.client.Set(ctx, sessionKey, sessionData, 24*time.Hour).Err(); err != nil {
		return err
	}

	// 添加到用户会话列表
	userSessionsKey := fmt.Sprintf("user_sessions:%s", session.UserID)
	return r.client.ZAdd(ctx, userSessionsKey, &redis.Z{
		Score:  float64(session.CreatedAt.Unix()),
		Member: session.ID,
	}).Err()
}

func (r *RedisService) GetSession(ctx context.Context, sessionID string) (*Session, error) {
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

func (r *RedisService) GetUserSessions(ctx context.Context, userID string) ([]*Session, error) {
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

func (r *RedisService) Close() error {
	return r.client.Close()
}
