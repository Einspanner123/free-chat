package mq

// type Topic string
// type Tag string

// func (t Topic) String() string {
// 	return string(t)
// }
// func (t Tag) String() string {
// 	return string(t)
// }

const (
	TopicPersistence = "persist_topic"
	TagSaveMessage   = "save_message"
	TagSaveSession   = "save_session"
	TagStreamToken   = "stream_token"
)
