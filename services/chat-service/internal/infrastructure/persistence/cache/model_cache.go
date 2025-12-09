package cache

import (
	"context"
	"errors"
	"fmt"

	"github.com/go-redis/redis/v8"
)

var (
	ErrNoInstanceProvided = errors.New("no instances provided")
	ErrFindBestInstance   = errors.New("failed to select best instance")
)
var selectBestFromListScript = redis.NewScript(`
	local key = KEYS[1]
	local best_member = nil
	local min_score = -1

	for i, member in ipairs(ARGV) do
		local score = redis.call("ZSCORE", key, member)
		if not score then
			redis.call("ZADD", key, 1, member)
			return member
		else
			score = tonumber(score)
		end
		
		if score == 0 then
			redis.call("ZINCRBY", key, 1, member)
			return member
		end

		if min_score == -1 or score < min_score then
			min_score = score
			best_member = member
		end
	end

	if best_member then
		redis.call("ZINCRBY", key, 1, best_member)
		return best_member
	end
	return nil
`)

func (r *RedisCache) modelKey(modelName string) string {
	return fmt.Sprintf("llm_models:%s", modelName)
}

func (r *RedisCache) SelectAndRegister(ctx context.Context, modelName string, instances []string) (string, error) {
	if len(instances) == 0 {
		return "", ErrNoInstanceProvided
	}
	key := r.modelKey(modelName)
	// Convert instances to []interface{} for args
	args := make([]interface{}, len(instances))
	for i, v := range instances {
		args[i] = v
	}

	res, err := selectBestFromListScript.Run(ctx, r.client, []string{key}, args...).Result()
	if err != nil && err != redis.Nil {
		return "", err
	}
	if res == nil {
		return "", ErrFindBestInstance
	}
	return res.(string), nil
}

func (r *RedisCache) DecrementTaskCount(ctx context.Context, modelName, address string) error {
	key := r.modelKey(modelName)
	return r.client.ZIncrBy(ctx, key, -1, address).Err()
}
