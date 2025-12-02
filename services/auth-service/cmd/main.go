package main

import (
	"fmt"
	"free-chat/config"
	store "free-chat/infra/storage"
	authpb "free-chat/pkg/proto/auth"
	"free-chat/pkg/registry"
	"free-chat/services/auth-service/internal/application"
	"free-chat/services/auth-service/internal/infrastructure/persistence"
	"free-chat/services/auth-service/internal/infrastructure/security"
	handler "free-chat/services/auth-service/internal/interfaces"
	"log"
	"net"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
)

func main() {
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}
	serviceName := cfg.Auth.ServerName
	grpcPort := cfg.Auth.GRPCPort
	var localIP string
	if localIP, err = registry.GetLocalIP(); err != nil {
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
	var serviceManager *registry.ServiceManager
	if serviceManager, err = registry.NewServiceManager(consulCfg, serviceCfg); err != nil {
		log.Fatalf("创建服务管理器失败: %v", err)
	}
	// 创建数据库连接
	var db *store.Postgres
	if db, err = store.NewPostgresConn(store.GetURL(&cfg.Postgres)); err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()
	// 创建数据库表
	// 注意：这里需要确保 persistence 层定义的 UserEntity 能被 AutoMigrate
	// 为了简化，我们这里先通过 persistence 层手动 migrate 或者复用 store 包的逻辑
	// 理想情况下，Migration 应该是独立的一步，或者由 Infrastructure 层负责
	if err = db.AutoMigrate(&persistence.UserEntity{}); err != nil {
		log.Fatalf("Failed to create tables: %v", err)
	}

	// 初始化依赖 (Infrastructure Layer)
	// 将 store.DB (GORM) 传递给 persistence
	userRepo := persistence.NewUserRepository(db.DB)
	passwordService := security.NewBcryptService()
	jwtService := security.NewJWTService(cfg.Auth.JwtSecret, cfg.Auth.Expire_Access_H, cfg.Auth.Expire_Refresh_H)

	// 初始化应用服务 (Application Layer)
	authService := application.NewAuthService(userRepo, passwordService, jwtService)

	// 初始化接口层 (Interface/Handler Layer)
	authHandler := handler.NewAuthHandler(authService)

	// 创建并启动gRPC服务器
	var lis net.Listener
	if lis, err = net.Listen("tcp", fmt.Sprintf(":%d", grpcPort)); err != nil {
		log.Fatalf("监听失败: %v", err)
	}
	grpcServer := grpc.NewServer()
	authpb.RegisterAuthServiceServer(grpcServer, authHandler)
	// 注册gRPC健康检查服务
	healthServer := health.NewServer()
	healthServer.SetServingStatus(serviceName, grpc_health_v1.HealthCheckResponse_SERVING)
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)

	// 注册到 Consul
	serviceManager.Start()
	// 启动gRPC服务器
	log.Printf("Auth gRPC 服务启动: %d", grpcPort)
	if err = grpcServer.Serve(lis); err != nil {
		log.Fatal(err)
	}
}
