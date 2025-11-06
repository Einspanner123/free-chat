package main

import (
	"free-chat/cmd/gateway/internal/handler"
	"free-chat/cmd/gateway/internal/middleware"
	"free-chat/shared/config"
	"free-chat/shared/registry"
	"log"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
)

func main() {
	cfg := config.LoadConfig("gateway")

	consulCfg := registry.ConsulConfig{
		Address:    cfg.Consul.Address,
		Scheme:     cfg.Consul.Scheme,
		Datacenter: cfg.Consul.Datacenter,
	}
	registry, err := registry.NewConsulRegistry(&consulCfg)
	if err != nil {
		log.Fatalf("注册Consul时出错: %v", err)
	}
	serviceCfg := registry.ServiceConfig{
		ID: registry.GenerateServiceID()
	}

	// consulClient, err := consul.NewClient(&cfg.Consul)
	if err != nil {
		log.Fatalf("初始化Consul客户端失败: %v", err)
	}
	redisClient := redis.NewClient(&redis.Options{
		Addr: cfg.Redis.Address + ":" + cfg.Redis.Port,
	})
	// set router
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
