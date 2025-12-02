package mq

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/apache/rocketmq-client-go/v2"
	"github.com/apache/rocketmq-client-go/v2/primitive"
)

const (
	TopicChat      = "chat_topic"
	TagSaveMsg     = "save_message"
	TagSaveSession = "save_session"
)

type Producer struct {
	client rocketmq.Producer
}

func NewProducer(client rocketmq.Producer) *Producer {
	return &Producer{client: client}
}

func (p *Producer) SendSaveMessageEvent(msg interface{}) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("converting msg: %w", err)
	}
	msg = primitive.NewMessage(TopicChat, data)
	msg.WithTag(TagSaveMsg)

	_, err = p.client.SendSync(context.Background(), msg)
	return err
}

func (p *Producer) SendSaveSessionEvent(session interface{}) error {
	data, _ := json.Marshal(session)
	msg := primitive.NewMessage(TopicChat, data)
	msg.WithTag(TagSaveSession)

	_, err := p.client.SendSync(context.Background(), msg)
	return err
}
