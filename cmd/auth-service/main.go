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

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
)

func main() {
	cfg := config.LoadConfig("auth-service")
	serviceName := cfg.Auth.ServerName
	grpcPort := cfg.Auth.GRPCPort
	var localIP string
	var err error
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
	var db *store.DB
	if db, err = store.NewConnection(store.GetURL(&cfg.Postgres)); err != nil {
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
