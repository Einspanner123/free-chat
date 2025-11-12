package main

import (
	"fmt"
	"free-chat/shared/config"
	"free-chat/shared/registry"
	"log"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
)

func main() {
	cfg := config.LoadConfig("chat-service")
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

	r := gin.New()
	r.Use(gin.Logger(), gin.Recovery())
	chat := r.Group("chat")
	{
		chat.GET("/health", func(c *gin.Context) {
			c.JSON(200, gin.H{
				"status":    "healthy",
				"service":   serviceName,
				"timestamp": time.Now(),
			})
		})
	}
	service.Start()
}
