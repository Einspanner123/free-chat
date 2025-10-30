package main

import (
	"free-chat/services/gateway/internal/consul"
	"free-chat/services/gateway/internal/handler"
	"free-chat/services/gateway/internal/middleware"
	"free-chat/shared/config"
	"log"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
)

func main() {
	cfg := config.LoadConfig("gateway")
	consulClient, err := consul.NewClient(&cfg.Consul)
	if err != nil {
		log.Fatalf("初始化Consul客户端失败: %v", err)
	}
	redisClient := redis.NewClient(&redis.Options{
		Addr: cfg.Redis.Address + ":" + cfg.Redis.Port,
	})
	r := gin.Default()
	r.SetTrustedProxies([]string{"127.0.0.1", "192.168.31.255"})
	r.Use(gin.Logger(), gin.Recovery())
	r.Use(middleware.RateLimit(redisClient, cfg.Redis.RateLimitQPS))

	api := r.Group("/api")
	{
		auth := api.Group("/")
		auth.Use(middleware.JwtAuth(cfg.Auth.JwtSecret))
		{
			auth.POST("/chat", handler.ChatHandler(consulClient, cfg.Chat.ServerName))
		}
	}

	log.Printf("网关服务启动, 监听服务 %s", cfg.Port)
	log.Fatal(r.Run(":" + cfg.Port))
}
