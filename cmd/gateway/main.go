package main

import (
	"fmt"
	"free-chat/cmd/gateway/internal/middleware"
	"free-chat/shared/config"
	"free-chat/shared/registry"
	"log"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
)

func main() {
	cfg := config.LoadConfig("gateway")
	serviceName := cfg.ServerName
	servicePort, _ := strconv.Atoi(cfg.Port)
	localIP, err := registry.GetLocalIP()
	if err != nil {
		log.Fatalf("获取本机IP失败: %v", err)
	}
	consulCfg := &registry.ConsulConfig{
		Address:    cfg.Consul.Address,
		Scheme:     cfg.Consul.Scheme,
		Datacenter: cfg.Consul.Datacenter,
	}
	consul, err := registry.NewConsulRegistry(consulCfg)
	if err != nil {
		log.Fatalf("注册Consul时出错: %v", err)
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
			DeregisterCriticalServiceAfter: 30 * time.Second,
		},
	}
	service, err := registry.NewServiceManager(consulCfg, serviceCfg)
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
	// r.Use(middleware.JwtAuth(consul, "auth-service"))
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":    "healthy",
			"service":   serviceName,
			"timestamp": time.Now(),
		})
	})
	api := r.Group("/api/v1")
	{
		api.POST("/auth", func(c *gin.Context) {

		})
	}

	service.Start()
	log.Printf("网关服务启动, 监听服务 %s", cfg.Port)
	log.Fatal(r.Run(":" + cfg.Port))
}
