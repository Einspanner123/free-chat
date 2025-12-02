package handler

// import (
// 	"context"
// 	"io"
// 	"log"
// 	"strings"

// 	"free-chat/cmd/chat-service/internal/service"
// 	"free-chat/cmd/chat-service/internal/store"
// 	chatpb "free-chat/shared/proto/chat"

// 	"github.com/google/uuid"

// 	"google.golang.org/grpc/codes"
// 	"google.golang.org/grpc/status"
// )

// type ChatHandler struct {
// 	chatpb.UnimplementedChatServiceServer
// 	// redis   *service.RedisService
// 	llmClnt *service.LLMClient
// 	history *store.HistoryRepository
// }

// func NewChatHandler(llmClient *service.LLMClient, history *store.HistoryRepository) *ChatHandler {
// 	return &ChatHandler{
// 		llmClnt: llmClient,
// 		history: history,
// 	}
// }

// func (h *ChatHandler) StreamChat(req *chatpb.ChatRequest, chatStream chatpb.ChatService_StreamChatServer) error {
// 	// 验证请求
// 	if req.UserId == "" || req.Message == "" {
// 		log.Printf("req.user_id={%s}, req.message={%s}", req.UserId, req.Message)
// 		return status.Errorf(codes.InvalidArgument, "user_id and message are required")
// 	}

// 	// 如果没有session_id，创建新会话
// 	sessionID := req.SessionId
// 	if sessionID == "" {
// 		sessionID = uuid.New().String()
// 	}

// 	if h.history != nil {
// 		if _, err := h.history.CreateHistory(req.UserId, sessionID, req.Message, "user"); err != nil {
// 			log.Printf("Failed to persist user message: %v", err)
// 		}
// 	}

// 	llmStream, err := h.llmClnt.StreamInference(context.Background(), sessionID, req.Message, req.LlmService)
// 	if err != nil {
// 		log.Printf("Failed to get response from LLM: %v", err)
// 		return status.Errorf(codes.Internal, "failed to get response from LLM: %v", err)
// 	}
// 	var fullResp strings.Builder
// 	for {
// 		respChunk, err := llmStream.Recv()
// 		if err == io.EOF {
// 			break
// 		} else if err != nil {
// 			log.Printf("Error receiving from LLM stream: %v", err)
// 			return status.Errorf(codes.Internal, "error receiving from LLM stream: %v", err)
// 		}
// 		// 发送流式响应
// 		if err := chatStream.Send(&chatpb.ChatResponse{
// 			SessionId:       sessionID,
// 			Token:           respChunk.Chunk,
// 			IsFinished:      respChunk.IsFinished,
// 			Error:           respChunk.Error,
// 			GeneratedTokens: respChunk.GeneratedTokens,
// 		}); err != nil {
// 			return err
// 		}
// 		fullResp.WriteString(respChunk.Chunk)
// 	}

// 	if h.history != nil {
// 		if _, err := h.history.CreateHistory(req.UserId, sessionID, fullResp.String(), "assistant"); err != nil {
// 			log.Printf("Failed to persist assistant message: %v", err)
// 		}
// 	}

// 	return nil
// }

// func (h *ChatHandler) GetChatHistory(ctx context.Context, req *chatpb.HistoryRequest) (*chatpb.HistoryResponse, error) {
// 	if req.UserId == "" {
// 		return nil, status.Errorf(codes.InvalidArgument, "user_id is required")
// 	}
// 	if h.history == nil {
// 		return nil, status.Errorf(codes.Internal, "history repository not initialized")
// 	}
// 	var items []*store.History
// 	var err error
// 	if req.SessionId != "" {
// 		items, err = h.history.GetHistoryBySession(req.SessionId, int(req.Limit), int(req.Offset))
// 	} else {
// 		items, err = h.history.GetHistoryByUserID(req.UserId, int(req.Limit), int(req.Offset))
// 	}
// 	if err != nil {
// 		return nil, status.Errorf(codes.Internal, "failed to get history")
// 	}
// 	msgs := make([]*chatpb.ChatMessage, 0, len(items))
// 	for _, it := range items {
// 		msgs = append(msgs, &chatpb.ChatMessage{
// 			SessionId: it.SessionID,
// 			Role:      it.Role,
// 			Content:   it.Content,
// 			Timestamp: it.CreatedAt.Unix(),
// 		})
// 	}
// 	return &chatpb.HistoryResponse{Messages: msgs, Total: int32(len(msgs))}, nil
// }

// func (h *ChatHandler) CreateSession(ctx context.Context, req *chatpb.CreateSessionRequest) (*chatpb.CreateSessionResponse, error) {
// 	if req.UserId == "" {
// 		return &chatpb.CreateSessionResponse{Success: false, Message: "user_id is required"}, nil
// 	}
// 	sid := uuid.New().String()
// 	return &chatpb.CreateSessionResponse{Success: true, SessionId: sid, Message: "session created"}, nil
// }

// func (h *ChatHandler) DeleteSession(ctx context.Context, req *chatpb.DeleteSessionRequest) (*chatpb.DeleteSessionResponse, error) {
// 	if req.UserId == "" || req.SessionId == "" {
// 		return &chatpb.DeleteSessionResponse{Success: false, Message: "user_id and session_id are required"}, nil
// 	}
// 	if h.history != nil {
// 		if err := h.history.DeleteBySession(req.SessionId, req.UserId); err != nil {
// 			log.Printf("Failed to delete session history: %v", err)
// 		}
// 	}
// 	return &chatpb.DeleteSessionResponse{Success: true, Message: "session deleted"}, nil
// }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// func (h *ChatHandler) GetChatHistory(ctx context.Context, req *chatpb.HistoryRequest) (*chatpb.HistoryResponse, error) {
// 	if req.UserId == "" {
// 		return nil, status.Errorf(codes.InvalidArgument, "user_id is required")
// 	}

// 	var messages []*service.Message
// 	var err error

// 	if req.SessionId != "" {
// 		// 获取特定会话的消息
// 		messages, err = h.redis.GetSessionMessages(ctx, req.SessionId)
// 	} else {
// 		// 获取用户的所有会话
// 		sessions, err := h.redis.GetUserSessions(ctx, req.UserId)
// 		if err != nil {
// 			return nil, status.Errorf(codes.Internal, "failed to get user sessions")
// 		}

// 		// 获取所有会话的消息（简化实现）
// 		for _, session := range sessions {
// 			sessionMessages, err := h.redis.GetSessionMessages(ctx, session.ID)
// 			if err != nil {
// 				continue
// 			}
// 			messages = append(messages, sessionMessages...)
// 		}
// 	}

// 	if err != nil {
// 		return nil, status.Errorf(codes.Internal, "failed to get messages")
// 	}

// 	// 转换为protobuf格式
// 	chatMessages := make([]*chatpb.ChatMessage, 0, len(messages))
// 	for _, msg := range messages {
// 		chatMessages = append(chatMessages, &chatpb.ChatMessage{
// 			SessionId: msg.SessionID,
// 			Role:      msg.Role,
// 			Content:   msg.Content,
// 			Timestamp: msg.Timestamp.Unix(),
// 		})
// 	}

// 	// 应用分页
// 	start := int(req.Offset)
// 	end := start + int(req.Limit)
// 	if req.Limit == 0 {
// 		end = len(chatMessages)
// 	}
// 	if start > len(chatMessages) {
// 		start = len(chatMessages)
// 	}
// 	if end > len(chatMessages) {
// 		end = len(chatMessages)
// 	}

// 	return &chatpb.HistoryResponse{
// 		Messages: chatMessages[start:end],
// 		Total:    int32(len(chatMessages)),
// 	}, nil
// }

// func (h *ChatHandler) CreateSession(ctx context.Context, req *chatpb.CreateSessionRequest) (*chatpb.CreateSessionResponse, error) {
// 	if req.UserId == "" {
// 		return &chatpb.CreateSessionResponse{
// 			Success: false,
// 			Message: "user_id is required",
// 		}, nil
// 	}

// 	sessionID := uuid.New().String()
// 	title := req.Title
// 	if title == "" {
// 		title = "新对话"
// 	}

// 	session := &service.Session{
// 		ID:        sessionID,
// 		UserID:    req.UserId,
// 		Title:     title,
// 		CreatedAt: time.Now(),
// 		UpdatedAt: time.Now(),
// 	}

// 	if err := h.redis.CreateSession(ctx, session); err != nil {
// 		log.Printf("Failed to create session: %v", err)
// 		return &chatpb.CreateSessionResponse{
// 			Success: false,
// 			Message: "failed to create session",
// 		}, nil
// 	}

// 	return &chatpb.CreateSessionResponse{
// 		Success:   true,
// 		SessionId: sessionID,
// 		Message:   "session created successfully",
// 	}, nil
// }

// func (h *ChatHandler) DeleteSession(ctx context.Context, req *chatpb.DeleteSessionRequest) (*chatpb.DeleteSessionResponse, error) {
// 	if req.UserId == "" || req.SessionId == "" {
// 		return &chatpb.DeleteSessionResponse{
// 			Success: false,
// 			Message: "user_id and session_id are required",
// 		}, nil
// 	}

// 	if err := h.redis.DeleteSession(ctx, req.SessionId, req.UserId); err != nil {
// 		log.Printf("Failed to delete session: %v", err)
// 		return &chatpb.DeleteSessionResponse{
// 			Success: false,
// 			Message: "failed to delete session",
// 		}, nil
// 	}

// 	return &chatpb.DeleteSessionResponse{
// 		Success: true,
// 		Message: "session deleted successfully",
// 	}, nil
// }

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
