package main

import (
	"fmt"
	"free-chat/config"
	"free-chat/infra/registry"
	"free-chat/services/api-gateway/internal/handler"
	"free-chat/services/api-gateway/internal/middleware"
	"log"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
)

func main() {
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}
	serviceName := cfg.ServerName
	servicePort := cfg.Port
	localIP, err := registry.GetLocalIP()
	if err != nil {
		log.Fatalf("获取本机IP失败: %v", err)
	}
	consulCfg := &registry.ConsulConfig{
		Address:    cfg.Consul.Address,
		Scheme:     cfg.Consul.Scheme,
		Datacenter: cfg.Consul.Datacenter,
	}
	serviceCfg := &registry.ServiceConfig{
		ID:      registry.GenerateServiceID(serviceName, servicePort),
		Name:    serviceName,
		Tags:    []string{serviceName, "api", "v1"},
		Address: localIP,
		Port:    servicePort,
		HealthCheck: &registry.HealthCheck{
			HTTP:                           fmt.Sprintf("http://%s:%d/health", localIP, servicePort),
			Interval:                       10 * time.Second,
			Timeout:                        3 * time.Second,
			DeregisterCriticalServiceAfter: 1 * time.Minute,
		},
	}
	serviceManager, err := registry.NewServiceManager(consulCfg, serviceCfg)
	if err != nil {
		log.Fatalf("初始化Consul客户端失败: %v", err)
	}
	redisClient := redis.NewClient(&redis.Options{
		Addr: fmt.Sprintf("%s:%d", cfg.Redis.Address, cfg.Redis.Port),
	})

	r := gin.Default()
	r.SetTrustedProxies([]string{
		"127.0.0.1/32",
		"192.168.31.0/24",
		"172.20.0.0/16",
	})
	r.Use(middleware.RateLimit(redisClient, cfg.Redis.RateLimitQPS))
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":    "healthy",
			"service":   serviceName,
			"timestamp": time.Now(),
		})
	})

	api := r.Group("/api/v1")
	{
		// 认证相关路由
		auth := api.Group("/auth")
		{
			authHandler := handler.NewAuthHandler(serviceManager, cfg.Auth.ServerName)
			auth.POST("/login", authHandler.Login)
			auth.POST("/register", authHandler.Register)
			auth.POST("/refresh", authHandler.RefreshToken)
		}

		// 聊天相关路由（需要认证）
		chat := api.Group("/chat")
		chat.Use(middleware.JwtAuth(cfg.Auth.JwtSecret))
		{
			chatHandler := handler.NewChatHandler(serviceManager, cfg.Chat.ServerName, cfg.LLM.Name)
			chat.POST("/sessions", chatHandler.CreateSession)
			chat.GET("/sessions/history", chatHandler.GetHistory)
			chat.DELETE("/sessions", chatHandler.DeleteSession)
			chat.POST("/sessions/messages", chatHandler.StreamChat)
			chat.POST("/sessions/stream", chatHandler.StreamChat)
		}
	}

	serviceManager.Start()
	log.Printf("网关服务启动, 监听服务 %d", cfg.Port)
	log.Fatal(r.Run(fmt.Sprintf(":%d", servicePort)))
}
