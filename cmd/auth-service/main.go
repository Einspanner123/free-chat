package main

import (
	"fmt"
	"free-chat/cmd/auth-service/internal/handler"
	"free-chat/cmd/auth-service/internal/service"
	"free-chat/cmd/auth-service/internal/store"
	"free-chat/shared/config"
	authpb "free-chat/shared/proto/auth"
	"free-chat/shared/registry"
	"log"
	"net"
	"time"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
)

func main() {
	cfg := config.LoadConfig("auth-service")
	serviceName := cfg.Auth.ServerName
	servicePort := 8082
	grpcPort := cfg.Auth.GRPCPort
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
		ID:      registry.GenerateServiceID(serviceName, grpcPort),
		Name:    serviceName,
		Tags:    []string{serviceName, "api", "v1"},
		Address: localIP,
		Port:    grpcPort,
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
	// 注册到 Consul
	serviceManager.Start()
	// 创建数据库连接
	db, err := store.NewConnection(store.GetURL(&cfg.Postgres))
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()
	// 创建数据库表
	if err = db.CreateTables(); err != nil {
		log.Fatalf("Failed to create tables: %v", err)
	}

	// 初始化服务
	userRepo := store.NewUserRepository(db)
	jwtService := service.NewJWTService(cfg.Auth.JwtSecret, cfg.Auth.Expire_Access_H, cfg.Auth.Expire_Refresh_H)
	authHandler := handler.NewAuthHandler(userRepo, jwtService)

	// 创建并启动gRPC服务器
	go func() {
		lis, err := net.Listen("tcp", fmt.Sprintf(":%d", grpcPort))
		if err != nil {
			log.Fatalf("监听失败: %v", err)
		}
		grpcServer := grpc.NewServer()
		authpb.RegisterAuthServiceServer(grpcServer, authHandler)

		log.Printf("Auth gRPC 服务启动: %d", grpcPort)
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatal(err)
		}
	}()

	// 启动Gin健康检查服务
	r := gin.Default()
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":    "healthy",
			"service":   serviceName,
			"timestamp": time.Now(),
		})
	})
	port := fmt.Sprintf(":%d", servicePort)
	log.Printf("健康检查服务启动，监听端口: %s", port)
	if err := r.Run(port); err != nil {
		log.Fatal(err)
	}
}
