package main

import (
	"fmt"
	"free-chat/shared/config"
	"free-chat/shared/registry"
	"free-chat/shared/service"
	"log"
	"time"

	"github.com/gin-gonic/gin"
)

func main() {
	serviceName := "auth-service"
	servicePort := 8082
	localIP, err := registry.GetLocalIP()
	if err != nil {
		log.Fatalf("获取本机IP失败: %v", err)
	}
	cfg := config.LoadConfig(serviceName)
	consulCfg := &registry.ConsulConfig{
		Address:    cfg.Consul.Address,
		Scheme:     cfg.Consul.Scheme,
		Datacenter: cfg.Consul.Datacenter,
	}
	serviceCfg := &registry.ServiceConfig{
		ID:      registry.GenerateServiceID(serviceName, servicePort),
		Name:    serviceName,
		Tags:    []string{"order", "api", "v1"},
		Address: localIP,
		Port:    servicePort,
		HealthCheck: &registry.HealthCheck{
			HTTP:                           fmt.Sprintf("http://%s:%d/health", localIP, servicePort),
			Interval:                       10 * time.Second,
			Timeout:                        3 * time.Second,
			DeregisterCriticalServiceAfter: 30 * time.Second,
		},
	}
	serviceManager, err := service.NewServiceManager(consulCfg, serviceCfg)
	if err != nil {
		log.Fatalf("创建服务管理器失败: %v", err)
	}
	serviceManager.Start()
	r := gin.Default()

}
