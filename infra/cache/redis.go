// infra/cache/redis.go
package cache

import (
	"context"
	"fmt"
	"free-chat/config"
	"math/rand"
	"time"

	"github.com/go-redis/redis/v8"
)

type RedisCache struct {
	client *redis.Client
	prefix string
}

func NewRedisCache(cfg *config.RedisConfig, prefix string) (*RedisCache, error) {
	client := redis.NewClient(&redis.Options{
		Addr:         fmt.Sprintf("%s:%d", cfg.Address, cfg.Port),
		Password:     cfg.Password,
		DB:           cfg.Database,
		DialTimeout:  cfg.DialTimeout,
		ReadTimeout:  cfg.WriteTimeout,
		WriteTimeout: cfg.ReadTimeout,
		MaxRetries:   cfg.MaxRetries,
		PoolSize:     cfg.PoolSize,
		MinIdleConns: cfg.MinIdleConns,
	})
	err := client.Ping(context.Background()).Err()
	if err != nil {
		return nil, fmt.Errorf("ping redis: %w", err)
	}
	redisClnt := RedisCache{
		client: client,
		prefix: prefix,
	}
	return &redisClnt, nil
}

func (r *RedisCache) GetWithProtection(ctx context.Context, key string, loader func() ([]byte, error)) ([]byte, error) {
	fullKey := r.prefix + key

	data, err := r.client.Get(ctx, fullKey).Bytes()
	if err == nil {
		return data, nil
	}
	if err != redis.Nil {
		return nil, err
	}
	lockKey := r.prefix + "lock:" + key

	// 尝试获取锁，最多重试5次 (50ms * 5 = 250ms)
	for range 5 {
		locked, err := r.client.SetNX(ctx, lockKey, "1", 10*time.Second).Result()
		if err != nil {
			// Redis错误，直接降级调用loader
			return loader()
		}

		if locked {
			// 获取锁成功
			defer r.client.Del(ctx, lockKey)

			// 加载数据
			data, err = loader()
			if err != nil {
				return nil, err
			}

			// 设置缓存，带随机过期时间防止雪崩
			baseTTL := 1 * time.Hour
			randomOffset := time.Duration(rand.Intn(600)) * time.Second
			ttl := baseTTL + randomOffset

			if err = r.client.Set(ctx, fullKey, data, ttl).Err(); err != nil {
				return data, err
			}

			return data, nil
		}

		// 未获取到锁，等待后重试读取
		time.Sleep(50 * time.Millisecond)
		data, err = r.client.Get(ctx, fullKey).Bytes()
		if err == nil {
			return data, nil
		}
	}

	return loader()
}

func (r *RedisCache) Close() error {
	return r.client.Close()
}
