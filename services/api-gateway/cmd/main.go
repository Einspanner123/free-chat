package main

import (
	"context"
	"fmt"
	"free-chat/config"
	"free-chat/pkg/registry"
	"free-chat/services/api-gateway/internal/handler"
	"free-chat/services/api-gateway/internal/middleware"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
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
	// Trust all proxies for development
	r.SetTrustedProxies(nil)
	r.Use(middleware.CORS())
	r.Use(middleware.RateLimit(redisClient, cfg.Redis.RateLimitQPS))
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":    "healthy",
			"service":   serviceName,
			"timestamp": time.Now(),
		})
	})

	api := r.Group("/api/v1")
	var authHandler *handler.AuthHandler
	var chatHandler *handler.ChatHandler
	{
		// 认证相关路由
		auth := api.Group("/auth")
		{
			authHandler = handler.NewAuthHandler(serviceManager, cfg.Auth.ServerName)
			auth.POST("/login", authHandler.Login)
			auth.POST("/register", authHandler.Register)
			auth.POST("/refresh", authHandler.RefreshToken)
		}

		// 聊天相关路由（需要认证）
		chat := api.Group("/chat")
		chat.Use(middleware.JwtAuth(cfg.Auth.JwtSecret))
		{
			chatHandler = handler.NewChatHandler(serviceManager, cfg.Chat.ServerName, cfg.LLM.Name)
			chat.POST("/sessions", chatHandler.CreateSession)
			chat.GET("/sessions/:sessionId/history", chatHandler.GetHistory)
			chat.DELETE("/sessions/:sessionId", chatHandler.DeleteSession)
			chat.POST("/sessions/messages", chatHandler.StreamChat)
			chat.POST("/sessions/stream", chatHandler.StreamChat)
		}
	}

	serviceManager.Start()
	log.Printf("网关服务启动, 监听服务 %d", cfg.Port)

	srv := &http.Server{
		Addr:    fmt.Sprintf(":%d", servicePort),
		Handler: r,
	}

	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("listen: %s\n", err)
		}
	}()

	// 优雅关闭
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatal("Server forced to shutdown: ", err)
	}

	if authHandler != nil {
		authHandler.Close()
	}
	if chatHandler != nil {
		chatHandler.Close()
	}

	if serviceManager != nil {
		serviceManager.Stop()
	}

	log.Println("Server exited")
}
