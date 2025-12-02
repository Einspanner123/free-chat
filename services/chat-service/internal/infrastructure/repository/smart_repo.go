package repository

import (
	"context"

	"free-chat/services/chat-service/internal/domain"
	"free-chat/services/chat-service/internal/infrastructure/cache"
	"free-chat/services/chat-service/internal/infrastructure/mq"

	"gorm.io/gorm"
)

// SmartChatRepository 负责协调 Redis, MQ 和 Postgres
type SmartRepository struct {
	rdb      *cache.RedisDB
	db       *gorm.DB
	producer *mq.Producer
}

func NewSmartRepository(rdb *cache.RedisCache, db *gorm.DB, producer *mq.Producer) domain.ChatRepository {
	return &SmartRepository{
		rdb:      rdb,
		db:       db,
		producer: producer,
	}
}

// SaveMessage 实现 "写缓存 + 投递MQ" 的策略
func (r *SmartChatRepository) SaveMessage(ctx context.Context, msg *domain.Message) error {
	// 1. 写入 Redis (List 或 ZSet)，保证前端刷新页面能立即看到
	// 设置较短的过期时间，或者作为热数据缓存
	if err := r.redis.SaveMessage(ctx, msg); err != nil {
		// 缓存失败不应阻断业务，但需要记录日志
		// log.Printf("redis save failed: %v", err)
	}

	// 2. 投递到 RocketMQ 进行异步持久化
	// 这是"不破坏用户空间"的关键，快速响应，后台慢慢落库
	return r.mq.PublishMessage(ctx, msg)
}

// GetHistory 实现 "读缓存 -> 降级读库" 的策略
func (r *SmartChatRepository) GetHistory(ctx context.Context, sessionID string, limit, offset int) ([]*domain.Message, error) {
	// 1. 尝试从 Redis 读取
	msgs, err := r.redis.GetHistory(ctx, sessionID, limit, offset)
	if err == nil && len(msgs) > 0 {
		return msgs, nil
	}

	// 2. 缓存未命中或不全，回源到 Postgres
	// 实际生产中可能需要做 Cache Aside 的回填逻辑
	return r.postgres.GetHistory(ctx, sessionID, limit, offset)
}

// CreateSession 同步写入 (因为 Session 创建频率低，且一致性要求高)
func (r *SmartChatRepository) CreateSession(ctx context.Context, session *domain.Session) error {
	if err := r.postgres.CreateSession(ctx, session); err != nil {
		return err
	}
	return r.redis.SaveSession(ctx, session)
}

// 省略其他方法的实现...
