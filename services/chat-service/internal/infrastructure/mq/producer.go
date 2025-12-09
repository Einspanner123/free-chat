package mq

import (
	"context"
	"encoding/json"
	"fmt"
	"free-chat/services/chat-service/internal/domain"
	"free-chat/services/chat-service/internal/infrastructure/persistence/model"

	rocketmq "github.com/apache/rocketmq-client-go/v2"
	"github.com/apache/rocketmq-client-go/v2/primitive"
)

type Producer struct{ client rocketmq.Producer }

func NewProducer(client rocketmq.Producer) *Producer {
	return &Producer{client: client}
}

func (p *Producer) SendSaveMessageEvent(msg *domain.Message) error {
	data, err := json.Marshal(model.ToMessageModel(msg))
	if err != nil {
		return fmt.Errorf("converting error: %w", err)
	}
	newMsg := primitive.NewMessage(TopicPersistence, data)
	newMsg.WithTag(TagSaveMessage)

	_, err = p.client.SendSync(context.Background(), newMsg)
	return err
}

func (p *Producer) SendSaveSessionEvent(session *domain.Session) error {
	data, err := json.Marshal(model.ToSessionModel(session))
	if err != nil {
		return fmt.Errorf("converting error: %w", err)
	}
	msg := primitive.NewMessage(TopicPersistence, data)
	msg.WithTag(TagSaveSession)

	_, err = p.client.SendSync(context.Background(), msg)
	return err
}
