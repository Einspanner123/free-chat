package queue

import (
	"github.com/google/uuid"
)

type Message struct {
	ID      string
	Payload []byte
}

// NewMessage 创建新消息
func NewMessage(payload []byte) Message {
	return Message{
		ID:      uuid.NewString(),
		Payload: payload,
	}
}
