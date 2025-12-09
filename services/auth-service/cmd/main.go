package main

import (
	"fmt"
	"free-chat/config"
	"free-chat/pkg/registry"
	"free-chat/services/auth-service/internal/application"
	"free-chat/services/auth-service/internal/domain"
	"free-chat/services/auth-service/internal/infrastructure/db"
	"free-chat/services/auth-service/internal/infrastructure/persistence"
	"free-chat/services/auth-service/internal/infrastructure/security"
	"free-chat/services/auth-service/internal/interfaces/grpc"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}
	serviceName := cfg.Auth.ServerName
	grpcPort := cfg.Auth.GRPCPort
	serverEndpoint := fmt.Sprintf("%s:%d", serviceName, grpcPort)
	localIP, err := registry.GetLocalIP()
	if err != nil {
		log.Fatalf("获取本机IP失败: %v", err)
	}
	// Consul配置
	consulCfg := &registry.ConsulConfig{
		Address:    cfg.Consul.Address,
		Scheme:     cfg.Consul.Scheme,
		Datacenter: cfg.Consul.Datacenter,
	}
	// Consul注册配置
	serviceCfg := &registry.ServiceConfig{
		ID:      registry.GenerateServiceID(serviceName, grpcPort),
		Name:    serviceName,
		Tags:    []string{serviceName, "api", "v1"},
		Address: localIP,
		Port:    grpcPort,
		HealthCheck: &registry.HealthCheck{
			Type:                           "grpc",
			GRPC:                           fmt.Sprintf("%s:%d", localIP, grpcPort),
			Interval:                       10 * time.Second,
			Timeout:                        3 * time.Second,
			DeregisterCriticalServiceAfter: 1 * time.Minute,
		},
	}
	//
	sm, err := registry.NewServiceManager(consulCfg, serviceCfg)
	if err != nil {
		log.Fatalf("创建服务管理器失败: %v", err)
	}
	// 创建数据库连接
	db, err := db.InitGorm(cfg.Postgres.Address)
	if err != nil {
		log.Fatal(err)
	}
	// 初始化依赖 (Infrastructure Layer)
	userRepo := persistence.NewUserRepository(db)
	passwordService := security.NewBcryptService()
	tokenService := security.NewJWTService(cfg.Auth.JwtSecret, cfg.Auth.Expire_Access_H, cfg.Auth.Expire_Refresh_H)

	// 初始化领域服务 (Domain Layer)
	userService := domain.NewUserService()

	// 初始化应用服务 (Application Layer)
	authService := application.NewAuthService(*userService, userRepo, tokenService, passwordService)

	// 初始化接口层 (Interface/Handler Layer)
	authHandler := grpc.NewAuthHandler(authService)
	authServer := grpc.NewAuthServer(serverEndpoint, serviceName, authHandler)

	if err := sm.Start(); err != nil {
		log.Fatalf("服务启动失败: %v", err)
	}

	// 启动 gRPC 服务
	go func() {
		log.Printf("Auth gRPC 服务启动: %d", grpcPort)
		if err = authServer.Serve(grpcPort); err != nil {
			log.Fatal(err)
		}
	}()

	// 优雅关闭
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit // 阻塞

	log.Println("Shutting down server...")

	sm.Stop()

	authServer.Stop()
	log.Println("Server exiting")
}
