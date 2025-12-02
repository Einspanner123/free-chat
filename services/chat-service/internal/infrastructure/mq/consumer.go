package mq

import (
	"context"
	"encoding/json"
	"log"

	"free-chat/services/chat-service/internal/domain"

	rocketmq "github.com/apache/rocketmq-client-go/v2"
	"github.com/apache/rocketmq-client-go/v2/consumer"
	"github.com/apache/rocketmq-client-go/v2/primitive"
	"gorm.io/gorm"
)

type ChatConsumer struct {
	client rocketmq.PushConsumer
	db     *gorm.DB
}

func NewChatConsumer(client rocketmq.PushConsumer, db *gorm.DB) *ChatConsumer {
	return &ChatConsumer{
		client: client,
		db:     db,
	}
}

func (c *ChatConsumer) Start() error {
	err := c.client.Subscribe(TopicChat, consumer.MessageSelector{}, c.handleMessage)
	if err != nil {
		return err
	}
	return c.client.Start()
}

func (c *ChatConsumer) handleMessage(ctx context.Context, msgs ...*primitive.MessageExt) (consumer.ConsumeResult, error) {
	for _, msg := range msgs {
		switch msg.GetTags() {
		case TagSaveMsg:
			var domainMsg domain.Message
			if err := json.Unmarshal(msg.Body, &domainMsg); err != nil {
				log.Printf("Unmarshal message error: %v", err)
				continue
			}
			// 持久化到 Postgres
			if err := c.db.Create(&domainMsg).Error; err != nil {
				log.Printf("DB Save Message Error: %v", err)
				return consumer.ConsumeRetryLater, err
			}

		case TagSaveSession:
			var session domain.Session
			if err := json.Unmarshal(msg.Body, &session); err != nil {
				log.Printf("Unmarshal session error: %v", err)
				continue
			}
			// Upsert Session
			if err := c.db.Save(&session).Error; err != nil {
				log.Printf("DB Save Session Error: %v", err)
				return consumer.ConsumeRetryLater, err
			}
		}
	}
	return consumer.ConsumeSuccess, nil
}
