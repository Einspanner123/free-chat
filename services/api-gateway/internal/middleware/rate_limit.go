package middleware

import (
	"log"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
)

const rateLimitLuaScript = `
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local bucket = redis.call('HMGET', key, 'tokens', 'updated_at')
local tokens = tonumber(bucket[1])
local updated_at = tonumber(bucket[2])

if tokens == nil or updated_at == nil then
    tokens = capacity
    updated_at = now
end

local elapsed = math.max(0, now - updated_at)
local added_tokens = elapsed * rate
tokens = math.min(capacity, tokens + added_tokens)

local allowed = 0
local retry_after = 0

if tokens >= requested then
    tokens = tokens - requested
    allowed = 1
else
    retry_after = (requested - tokens) / rate
end

redis.call('HMSET', key, 'tokens', tokens, 'updated_at', now)
redis.call('EXPIRE', key, 86400)

return {allowed, math.floor(tokens), math.ceil(retry_after)}
`

func RateLimit(redisClient *redis.Client, qps int) gin.HandlerFunc {
	return func(c *gin.Context) {
		ip := c.ClientIP()
		key := "rate_limit:" + ip

		// 使用令牌桶算法，容量为2*qps，令牌生成速率为qps个/秒
		capacity := 2 * qps
		rate := float64(qps)
		now := float64(time.Now().UnixNano()) / 1e9
		requested := 1 // 每个请求消耗1个令牌

		// 执行Lua脚本保证原子性
		result, err := redisClient.Eval(
			c.Request.Context(),
			rateLimitLuaScript,
			[]string{key},
			capacity, rate, now, requested,
		).Result()
		if err != nil {
			log.Printf("限流服务异常(已降级): %v", err)
			// Fail-Open: Redis挂了不能影响业务，直接放行
			c.Next()
			return
		}

		// 初始值
		allowed := int64(0)
		remaining := capacity
		retryAfter := 0 // 默认为0

		// Lua脚本保证返回整数，Go-Redis解析为int64
		if arr, ok := result.([]any); ok && len(arr) >= 3 {
			if v, ok := arr[0].(int64); ok {
				allowed = v
			}
			if v, ok := arr[1].(int64); ok {
				remaining = int(v)
			}
			if v, ok := arr[2].(int64); ok {
				retryAfter = int(v)
			}
		}

		if allowed == 0 {
			c.Header("X-RateLimit-Limit", strconv.Itoa(capacity))
			c.Header("Retry-After", strconv.Itoa(retryAfter))
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error": "请求过于频繁，请稍后再试",
			})
			c.Abort()
			return
		}

		c.Header("X-RateLimit-Limit", strconv.Itoa(capacity))
		c.Header("X-RateLimit-Remaining", strconv.Itoa(remaining))
		c.Next()
	}
}
