package main

import (
	"fmt"
	"free-chat/config"
	chatpb "free-chat/pkg/proto/chat"
	"free-chat/pkg/registry"
	"free-chat/services/chat-service/internal/application"
	"free-chat/services/chat-service/internal/infrastructure/adapter"
	"free-chat/services/chat-service/internal/infrastructure/mq"
	"free-chat/services/chat-service/internal/infrastructure/persistence/cache"
	"free-chat/services/chat-service/internal/infrastructure/persistence/repository"
	handler "free-chat/services/chat-service/internal/interfaces"
	"free-chat/services/chat-service/internal/store"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/apache/rocketmq-client-go/v2"
	"github.com/apache/rocketmq-client-go/v2/consumer"
	"github.com/apache/rocketmq-client-go/v2/primitive"
	"github.com/apache/rocketmq-client-go/v2/producer"
	"github.com/go-redis/redis/v8"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

func main() {
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

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

	// Initialize Redis
	redisClient := redis.NewClient(&redis.Options{
		Addr:         fmt.Sprintf("%s:%d", cfg.Redis.Address, cfg.Redis.Port),
		Password:     cfg.Redis.Password,
		DB:           cfg.Redis.Database,
		DialTimeout:  cfg.Redis.DialTimeout,
		ReadTimeout:  cfg.Redis.ReadTimeout,
		WriteTimeout: cfg.Redis.WriteTimeout,
		PoolSize:     cfg.Redis.PoolSize,
		MinIdleConns: cfg.Redis.MinIdleConns,
	})
	redisCache, err := cache.NewRedisCache(redisClient)
	if err != nil {
		log.Printf("Failed to initialize Redis cache: %v", err)
	}
	defer redisCache.Close()

	// Initialize RocketMQ Producer
	var mqProducer *mq.Producer
	if len(cfg.RocketMQ.NameServers) > 0 {
		p, err := rocketmq.NewProducer(
			producer.WithNsResolver(primitive.NewPassthroughResolver(cfg.RocketMQ.NameServers)),
			producer.WithRetry(cfg.RocketMQ.MaxRetries),
		)
		if err != nil {
			log.Printf("Failed to create RocketMQ producer: %v", err)
		} else {
			if err := p.Start(); err != nil {
				log.Printf("Failed to start RocketMQ producer: %v", err)
			} else {
				mqProducer = mq.NewProducer(p)
				defer func() {
					if err := p.Shutdown(); err != nil {
						log.Printf("Failed to shutdown RocketMQ producer: %v", err)
					}
				}()
			}
		}
	} else {
		log.Println("RocketMQ name servers not configured, skipping producer initialization")
	}

	pgUrl := store.GetURL(&cfg.Postgres)
	var msgRepo *repository.MessageRepository
	var sessionRepo *repository.SessionRepository

	if db, err := store.NewPostgresConn(pgUrl); err != nil {
		log.Printf("Postgres不可用，历史记录将不会持久化: %v", err)
	} else {
		if err := db.CreateTables(); err != nil {
			log.Printf("数据库迁移失败，历史记录将不会持久化: %v", err)
		} else {
			msgRepo = repository.NewMessageRepository(db.DB)
			sessionRepo = repository.NewSessionRepository(db.DB)
		}
	}

	// Initialize RocketMQ Consumer
	var mqConsumer *mq.Consumer
	if len(cfg.RocketMQ.NameServers) > 0 {
		c, err := rocketmq.NewPushConsumer(
			consumer.WithNsResolver(primitive.NewPassthroughResolver(cfg.RocketMQ.NameServers)),
			consumer.WithGroupName(cfg.RocketMQ.GroupName),
			consumer.WithRetry(cfg.RocketMQ.MaxRetries),
		)
		if err != nil {
			log.Printf("Failed to create RocketMQ consumer: %v", err)
		} else {
			mqConsumer = mq.NewConsumer(c, msgRepo, sessionRepo)
			// Subscribe to topics
			if err := mqConsumer.SubscribePersistence(); err != nil {
				log.Printf("Failed to subscribe persistence topic: %v", err)
			}

			// Start consumer
			if err := mqConsumer.Start(); err != nil {
				log.Printf("Failed to start RocketMQ consumer: %v", err)
			} else {
				log.Println("✅ RocketMQ Consumer started")
				defer func() {
					if err := c.Shutdown(); err != nil {
						log.Printf("Failed to shutdown RocketMQ consumer: %v", err)
					}
				}()
			}
		}
	}

	// Initialize Adapters
	chatRepoAdapter := adapter.NewChatRepositoryAdapter(redisCache, msgRepo, sessionRepo, mqProducer, nil)
	modelRepoAdapter := adapter.NewModelRepositoryAdapter(redisCache, svcMgr)
	llmClient := handler.NewLLMClient()

	// Initialize Application
	chatApp := application.NewChatService(chatRepoAdapter, modelRepoAdapter)

	// Initialize Handler
	chatHandler := handler.NewChatHandler(chatApp, llmClient)

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

	go func() {
		if err = grpcServer.Serve(lis); err != nil {
			log.Fatalf("Failed to serve: %v", err)
		}
	}()

	// Graceful Shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down server...")

	if svcMgr != nil {
		svcMgr.Stop()
	}

	grpcServer.GracefulStop()
	log.Printf("`%s` Server exited", cfg.Chat.ServerName)
}
