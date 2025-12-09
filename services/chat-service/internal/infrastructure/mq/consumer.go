package mq

import (
	"context"
	"encoding/json"
	"log"

	"free-chat/services/chat-service/internal/domain"
	"free-chat/services/chat-service/internal/infrastructure/persistence/repository"

	rocketmq "github.com/apache/rocketmq-client-go/v2"
	"github.com/apache/rocketmq-client-go/v2/consumer"
	"github.com/apache/rocketmq-client-go/v2/primitive"
)

type Consumer struct {
	client      rocketmq.PushConsumer
	msgRepo     *repository.MessageRepository
	sessionRepo *repository.SessionRepository
}

func NewConsumer(
	client rocketmq.PushConsumer,
	msgRepo *repository.MessageRepository,
	sessionRepo *repository.SessionRepository,
) *Consumer {
	return &Consumer{
		client:      client,
		msgRepo:     msgRepo,
		sessionRepo: sessionRepo,
	}
}

func (c *Consumer) Subscribe(
	topic string,
	handler func(context.Context, ...*primitive.MessageExt) (consumer.ConsumeResult, error),
) error {
	return c.client.Subscribe(topic, consumer.MessageSelector{}, handler)
}

func (c *Consumer) SubscribePersistence() error {
	return c.client.Subscribe(
		TopicPersistence,
		consumer.MessageSelector{},
		c.handlePersistenceMessage,
	)
}

func (c *Consumer) handlePersistenceMessage(ctx context.Context, msgs ...*primitive.MessageExt) (consumer.ConsumeResult, error) {
	for _, msg := range msgs {
		var err error
		switch msg.GetTags() {
		case TagSaveMessage:
			err = c.handleSaveMessage(ctx, msg.Body)
		case TagSaveSession:
			err = c.handleSaveSession(ctx, msg.Body)
		default:
			log.Printf("[WARN] unknown tag: %s", msg.GetTags())
			continue
		}

		if err != nil {
			log.Printf("[ERROR] handle message failed, will retry: %v", err)
			return consumer.ConsumeRetryLater, nil
		}
	}
	return consumer.ConsumeSuccess, nil
}

func (c *Consumer) handleSaveMessage(ctx context.Context, body []byte) error {
	var msg domain.Message
	if err := json.Unmarshal(body, &msg); err != nil {
		log.Printf("[ERROR] unmarshal message error: %v", err)
		return nil
	}

	if err := c.msgRepo.Save(ctx, &msg); err != nil {
		return err
	}
	log.Printf("[INFO] message persisted: %s", msg.ID)
	return nil
}

func (c *Consumer) handleSaveSession(ctx context.Context, body []byte) error {
	var session domain.Session
	if err := json.Unmarshal(body, &session); err != nil {
		log.Printf("[ERROR] unmarshal session error: %v", err)
		return nil
	}

	if err := c.sessionRepo.Save(ctx, &session); err != nil {
		return err
	}
	log.Printf("[INFO] session persisted: %s", session.ID)
	return nil
}

func (c *Consumer) Start() error {
	return c.client.Start()
}

func (c *Consumer) Shutdown() error {
	return c.client.Shutdown()
}
