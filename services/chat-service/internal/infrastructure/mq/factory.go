package mq

import (
	"context"
	"fmt"
	"log"
	"net"

	"free-chat/config"
	"free-chat/services/chat-service/internal/infrastructure/persistence/repository"

	rocketmq "github.com/apache/rocketmq-client-go/v2"
	"github.com/apache/rocketmq-client-go/v2/consumer"
	"github.com/apache/rocketmq-client-go/v2/primitive"
	"github.com/apache/rocketmq-client-go/v2/producer"
)

// InitProducer initializes the RocketMQ producer
func InitProducer(cfg *config.AppConfig) (*Producer, error) {
	resolvedNameServers := resolveNameServers(cfg.RocketMQ.NameServers)
	if len(resolvedNameServers) == 0 {
		log.Println("RocketMQ name servers not configured, skipping producer initialization")
		return nil, nil
	}

	p, err := rocketmq.NewProducer(
		producer.WithNsResolver(primitive.NewPassthroughResolver(resolvedNameServers)),
		producer.WithRetry(cfg.RocketMQ.MaxRetries),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create RocketMQ producer: %w", err)
	}

	if err := p.Start(); err != nil {
		return nil, fmt.Errorf("failed to start RocketMQ producer: %w", err)
	}

	// Send dummy message to create topic (autoCreateTopicEnable=true)
	dummyMsg := primitive.NewMessage(TopicPersistence, []byte("init"))
	if _, err := p.SendSync(context.Background(), dummyMsg); err != nil {
		log.Printf("Failed to send init message to %s: %v", TopicPersistence, err)
	} else {
		log.Printf("Initialized topic %s", TopicPersistence)
	}

	return NewProducer(p), nil
}

// InitConsumer initializes the RocketMQ consumer
func InitConsumer(cfg *config.AppConfig, msgRepo *repository.MessageRepository, sessionRepo *repository.SessionRepository) (*Consumer, error) {
	resolvedNameServers := resolveNameServers(cfg.RocketMQ.NameServers)
	if len(resolvedNameServers) == 0 {
		log.Println("RocketMQ name servers not configured, skipping consumer initialization")
		return nil, nil
	}

	c, err := rocketmq.NewPushConsumer(
		consumer.WithNsResolver(primitive.NewPassthroughResolver(resolvedNameServers)),
		consumer.WithGroupName(cfg.RocketMQ.ConsumerGroup),
		consumer.WithRetry(cfg.RocketMQ.MaxRetries),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create RocketMQ consumer: %w", err)
	}

	mqConsumer := NewConsumer(c, msgRepo, sessionRepo)

	// Subscribe to topics
	if err := mqConsumer.SubscribePersistence(); err != nil {
		return nil, fmt.Errorf("failed to subscribe persistence topic: %w", err)
	}

	// Start consumer
	if err := mqConsumer.Start(); err != nil {
		return nil, fmt.Errorf("failed to start RocketMQ consumer: %w", err)
	}

	log.Println("✅ RocketMQ Consumer started")
	return mqConsumer, nil
}

func resolveNameServers(servers []string) []string {
	var resolvedNameServers []string
	for _, addr := range servers {
		host, port, err := net.SplitHostPort(addr)
		if err != nil {
			log.Printf("Failed to split host port for %s: %v", addr, err)
			resolvedNameServers = append(resolvedNameServers, addr)
			continue
		}
		ips, err := net.LookupHost(host)
		if err != nil {
			log.Printf("Failed to lookup host %s: %v", host, err)
			resolvedNameServers = append(resolvedNameServers, addr)
			continue
		}
		if len(ips) > 0 {
			resolvedNameServers = append(resolvedNameServers, net.JoinHostPort(ips[0], port))
			log.Printf("Resolved %s to %s", addr, net.JoinHostPort(ips[0], port))
		} else {
			resolvedNameServers = append(resolvedNameServers, addr)
		}
	}
	return resolvedNameServers
}
