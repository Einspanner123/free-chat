package middleware

import (
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
)

func RateLimit(redisClient *redis.Client, qps int) gin.HandlerFunc {
	return func(c *gin.Context) {
		ip := c.ClientIP()
		key := "rate_limit:" + ip

		// 使用Redis实现令牌桶：1秒内最多qps个请求
		// 逻辑：incr计数，设置过期时间1秒，若计数>qps则限流
		ctx := c.Request.Context()
		count, err := redisClient.Incr(ctx, key).Result()
		if err != nil {
			log.Printf("限流服务出错: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "限流服务异常"})
			c.Abort()
			return
		}
		// 首次过期时间
		if count == 1 {
			redisClient.Expire(ctx, key, time.Second)
		}

		if count > int64(qps) {
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error": "请求过于频繁，请稍后再试",
				"qps":   qps,
			})
			c.Abort()
			return
		}
		c.Next()
	}
}
