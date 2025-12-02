package main

import (
	"fmt"
	"free-chat/cmd/chat-service/internal/handler"
	"free-chat/cmd/chat-service/internal/service"
	"free-chat/cmd/chat-service/internal/store"
	"free-chat/shared/config"
	chatpb "free-chat/shared/proto/chat"
	"free-chat/shared/registry"
	"log"
	"net"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

func main() {
	cfg := config.LoadConfig("chat-service")
	serviceName := cfg.ServerName
	grpcPort := cfg.Chat.GRPCPort
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
			Type:                           "grpc",
			GRPC:                           fmt.Sprintf("%s:%d", localIP, grpcPort),
			Interval:                       10 * time.Second,
			Timeout:                        3 * time.Second,
			DeregisterCriticalServiceAfter: 1 * time.Minute,
		},
	}
	var svcMgr *registry.ServiceManager
	if svcMgr, err = registry.NewServiceManager(consulCfg, serviceCfg); err != nil {
		log.Printf("初始化Consul客户端失败: %v", err)
	}

	// redisAddr := fmt.Sprintf("%s:%d", cfg.Redis.Address, cfg.Redis.Port)
	// var redis *service.RedisService
	// if redis, err = service.NewRedisService(redisAddr, cfg.Redis.Database); err != nil {
	// 	log.Fatalf("Failed to connect to Redis: %v", err)
	// }
	// defer redis.Close()
	llmClient := service.NewLLMClient(svcMgr)
	pgUrl := store.GetURL(&cfg.Postgres)
	var historyRepo *store.HistoryRepository
	if db, err := store.NewPostgresConn(pgUrl); err != nil {
		log.Printf("Postgres不可用，历史记录将不会持久化: %v", err)
	} else {
		if err := db.CreateTables(); err != nil {
			log.Printf("数据库迁移失败，历史记录将不会持久化: %v", err)
		} else {
			historyRepo = store.NewHistoryRepository(db)
		}
	}
	chatHandler := handler.NewChatHandler(llmClient, historyRepo)
	grpcServer := grpc.NewServer()
	chatpb.RegisterChatServiceServer(grpcServer, chatHandler)
	reflection.Register(grpcServer)
	// 注册健康检查服务
	healthServer := health.NewServer()
	healthServer.SetServingStatus(serviceName, grpc_health_v1.HealthCheckResponse_SERVING)
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)

	var lis net.Listener
	if lis, err = net.Listen("tcp", fmt.Sprintf(":%d", grpcPort)); err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}
	log.Printf("Chat service listening on port %d", grpcPort)

	if svcMgr != nil {
		svcMgr.Start()
	}

	if err = grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
