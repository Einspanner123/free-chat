package queue

import (
	"context"
	"fmt"

	rocketmq "github.com/apache/rocketmq-client-go/v2"
	"github.com/apache/rocketmq-client-go/v2/primitive"
	"github.com/apache/rocketmq-client-go/v2/producer"
)

type Producer struct {
	producer rocketmq.Producer
}

func NewProducer(nameServers []string, maxRetries int) (*Producer, error) {
	p, err := rocketmq.NewProducer(
		producer.WithNsResolver(primitive.NewPassthroughResolver(nameServers)),
		producer.WithRetry(maxRetries),
		producer.WithQueueSelector(producer.NewRoundRobinQueueSelector()),
	)
	if err != nil {
		return nil, fmt.Errorf("create producer: %w", err)
	}
	if err := p.Start(); err != nil {
		return nil, fmt.Errorf("start producer: %w", err)
	}
	return &Producer{producer: p}, nil
}

// Send 发送消息
func (p *Producer) Send(ctx context.Context, topic string, payload []byte) error {
	msg := &primitive.Message{
		Topic: topic,
		Body:  payload,
	}

	result, err := p.producer.SendSync(ctx, msg)
	if err != nil {
		return fmt.Errorf("send to %s: %w", topic, err)
	}

	if result.Status != primitive.SendOK {
		return fmt.Errorf("send failed: status=%d", result.Status)
	}
	return nil
}

func (p *Producer) Stop() error {
	return p.producer.Shutdown()
}
