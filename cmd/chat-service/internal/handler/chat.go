package handler

import (
	"context"
	"fmt"
	"log"
	"time"

	"free-chat/cmd/chat-service/internal/service"
	chatpb "free-chat/shared/proto/chat"

	"github.com/google/uuid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type ChatHandler struct {
	chatpb.UnimplementedChatServiceServer
	redisService *service.RedisService
}

func NewChatHandler(redisService *service.RedisService) *ChatHandler {
	return &ChatHandler{
		redisService: redisService,
	}
}

func (h *ChatHandler) StreamChat(req *chatpb.ChatRequest, stream chatpb.ChatService_StreamChatServer) error {
	// 验证请求
	if req.UserId == "" || req.Message == "" {
		return status.Errorf(codes.InvalidArgument, "user_id and message are required")
	}

	// 如果没有session_id，创建新会话
	sessionID := req.SessionId
	if sessionID == "" {
		sessionID = uuid.New().String()
		session := &service.Session{
			ID:        sessionID,
			UserID:    req.UserId,
			Title:     truncateString(req.Message, 50),
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		if err := h.redisService.CreateSession(context.Background(), session); err != nil {
			log.Printf("Failed to create session: %v", err)
			return status.Errorf(codes.Internal, "failed to create session")
		}
	}

	// 保存用户消息
	userMessage := &service.Message{
		ID:        uuid.New().String(),
		SessionID: sessionID,
		UserID:    req.UserId,
		Content:   req.Message,
		Role:      "user",
		Timestamp: time.Now(),
	}
	if err := h.redisService.SaveMessage(context.Background(), userMessage); err != nil {
		log.Printf("Failed to save user message: %v", err)
		return status.Errorf(codes.Internal, "failed to save message")
	}

	// 这里应该调用LLM推理服务，现在先模拟响应
	response := fmt.Sprintf("这是对消息 '%s' 的回复", req.Message)

	// 分块发送响应（模拟流式响应）
	words := []string{"这是", "对消息", fmt.Sprintf("'%s'", req.Message), "的", "回复"}
	for i, word := range words {
		if err := stream.Send(&chatpb.ChatResponse{
			SessionId:       sessionID,
			Token:           word,
			IsFinished:      i == len(words)-1,
			GeneratedTokens: int32(i + 1),
		}); err != nil {
			return err
		}
		time.Sleep(100 * time.Millisecond) // 模拟处理时间
	}

	// 保存助手消息
	assistantMessage := &service.Message{
		ID:        uuid.New().String(),
		SessionID: sessionID,
		UserID:    req.UserId,
		Content:   response,
		Role:      "assistant",
		Timestamp: time.Now(),
	}
	if err := h.redisService.SaveMessage(context.Background(), assistantMessage); err != nil {
		log.Printf("Failed to save assistant message: %v", err)
	}

	return nil
}

func (h *ChatHandler) GetChatHistory(ctx context.Context, req *chatpb.HistoryRequest) (*chatpb.HistoryResponse, error) {
	if req.UserId == "" {
		return nil, status.Errorf(codes.InvalidArgument, "user_id is required")
	}

	var messages []*service.Message
	var err error

	if req.SessionId != "" {
		// 获取特定会话的消息
		messages, err = h.redisService.GetSessionMessages(ctx, req.SessionId)
	} else {
		// 获取用户的所有会话
		sessions, err := h.redisService.GetUserSessions(ctx, req.UserId)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to get user sessions")
		}

		// 获取所有会话的消息（简化实现）
		for _, session := range sessions {
			sessionMessages, err := h.redisService.GetSessionMessages(ctx, session.ID)
			if err != nil {
				continue
			}
			messages = append(messages, sessionMessages...)
		}
	}

	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get messages")
	}

	// 转换为protobuf格式
	chatMessages := make([]*chatpb.ChatMessage, 0, len(messages))
	for _, msg := range messages {
		chatMessages = append(chatMessages, &chatpb.ChatMessage{
			Id:        msg.ID,
			SessionId: msg.SessionID,
			Role:      msg.Role,
			Content:   msg.Content,
			Timestamp: msg.Timestamp.Unix(),
		})
	}

	// 应用分页
	start := int(req.Offset)
	end := start + int(req.Limit)
	if req.Limit == 0 {
		end = len(chatMessages)
	}
	if start > len(chatMessages) {
		start = len(chatMessages)
	}
	if end > len(chatMessages) {
		end = len(chatMessages)
	}

	return &chatpb.HistoryResponse{
		Messages: chatMessages[start:end],
		Total:    int32(len(chatMessages)),
	}, nil
}

func (h *ChatHandler) CreateSession(ctx context.Context, req *chatpb.CreateSessionRequest) (*chatpb.CreateSessionResponse, error) {
	if req.UserId == "" {
		return &chatpb.CreateSessionResponse{
			Success: false,
			Message: "user_id is required",
		}, nil
	}

	sessionID := uuid.New().String()
	title := req.Title
	if title == "" {
		title = "新对话"
	}

	session := &service.Session{
		ID:        sessionID,
		UserID:    req.UserId,
		Title:     title,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	if err := h.redisService.CreateSession(ctx, session); err != nil {
		log.Printf("Failed to create session: %v", err)
		return &chatpb.CreateSessionResponse{
			Success: false,
			Message: "failed to create session",
		}, nil
	}

	return &chatpb.CreateSessionResponse{
		Success:   true,
		SessionId: sessionID,
		Message:   "session created successfully",
	}, nil
}

func (h *ChatHandler) DeleteSession(ctx context.Context, req *chatpb.DeleteSessionRequest) (*chatpb.DeleteSessionResponse, error) {
	if req.UserId == "" || req.SessionId == "" {
		return &chatpb.DeleteSessionResponse{
			Success: false,
			Message: "user_id and session_id are required",
		}, nil
	}

	if err := h.redisService.DeleteSession(ctx, req.SessionId, req.UserId); err != nil {
		log.Printf("Failed to delete session: %v", err)
		return &chatpb.DeleteSessionResponse{
			Success: false,
			Message: "failed to delete session",
		}, nil
	}

	return &chatpb.DeleteSessionResponse{
		Success: true,
		Message: "session deleted successfully",
	}, nil
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
