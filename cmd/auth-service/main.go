package main

import (
	"fmt"
	server "free-chat/cmd/auth-service/internal/grpc-server"
	"free-chat/cmd/auth-service/internal/service"
	"free-chat/cmd/auth-service/internal/store"
	"free-chat/shared/config"
	"free-chat/shared/proto/auth"
	"free-chat/shared/registry"
	"log"
	"net"
	"time"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
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

	serviceManager, err := registry.NewServiceManager(consulCfg, serviceCfg)
	if err != nil {
		log.Fatalf("创建服务管理器失败: %v", err)
	}
	serviceManager.Start()
	// 初始化存储与业务服务
	pg, err := store.NewPostgresStore(cfg.Postgres)
	if err != nil {
		log.Fatalf("数据库初始化失败: %v", err)
	}
	svc := service.NewAuthService(cfg.Auth, pg)
	// 启动 gRPC 认证服务
	go func() {
		lis, err := net.Listen("tcp", fmt.Sprintf(":%d", servicePort))
		if err != nil {
			log.Fatalf("监听失败: %v", err)
		}
		grpcServer := grpc.NewServer()
		auth.RegisterAuthServiceServer(grpcServer, &server.AuthGRPCServer{Svc: svc})
		log.Printf("Auth gRPC 服务启动: %d", servicePort)
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatal(err)
		}
	}()
	r := gin.Default()
	// 健康检查接口
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":    "healthy",
			"service":   serviceName,
			"timestamp": time.Now(),
		})
	})
	if err := r.Run(fmt.Sprintf(":%d", servicePort)); err != nil {
		log.Fatal(err)
	}
	// api := r.Group("/api/v1")
	// {
	// 	api.POST("/login", func(c *gin.Context) {

	// 	})
	// }
	// port := fmt.Sprintf(":%d", servicePort)
	// log.Printf("🚀 用户服务启动成功! 监听端口: %s", port)
	// log.Printf("📍 服务ID: %s", serviceConfig.ID)
	// log.Printf("🏥 健康检查: %s", serviceConfig.HealthCheck.HTTP)
	// log.Println("📋 API接口:")
	// log.Println("   GET  /health                    - 健康检查")
	// log.Println("   GET  /api/v1/users              - 获取用户列表")
	// log.Println("   GET  /api/v1/users/:id          - 获取用户详情")
	// log.Println("   GET  /api/v1/services/:name     - 服务发现")
	// log.Println("   GET  /api/v1/call/:service/*path - 调用其他服务")
	// if err := r.Run(port); err != nil {
	// 	log.Fatal("服务器启动失败: ", err)
	// }
}
