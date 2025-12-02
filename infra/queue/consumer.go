package queue

import (
	"context"
	"fmt"

	rocketmq "github.com/apache/rocketmq-client-go/v2"
	"github.com/apache/rocketmq-client-go/v2/consumer"
	"github.com/apache/rocketmq-client-go/v2/primitive"
)

type Consumer struct {
	consumer rocketmq.PushConsumer
}

func NewConsumer(nameServers []string, group string, model consumer.MessageModel) (*Consumer, error) {
	c, err := rocketmq.NewPushConsumer(
		consumer.WithNsResolver(primitive.NewPassthroughResolver(nameServers)),
		consumer.WithGroupName(group),
		consumer.WithConsumerModel(model),
	)
	if err != nil {
		return nil, fmt.Errorf("create consumer: %w", err)
	}

	return &Consumer{consumer: c}, nil
}

func (c *Consumer) Subscribe(topic string, handler func(context.Context, Message) error) error {
	wrappedHandler := func(ctx context.Context, msgs ...*primitive.MessageExt) (consumer.ConsumeResult, error) {
		for _, msg := range msgs {
			m := Message{
				ID:      msg.MsgId,
				Payload: msg.Body,
			}
			if err := handler(ctx, m); err != nil {
				return consumer.ConsumeRetryLater, err
			}
		}
		return consumer.ConsumeSuccess, nil
	}
	return c.consumer.Subscribe(topic, consumer.MessageSelector{}, wrappedHandler)
}

func (c *Consumer) Start() error {
	return c.consumer.Start()
}

func (c *Consumer) Stop() error {
	return c.consumer.Shutdown()
}
