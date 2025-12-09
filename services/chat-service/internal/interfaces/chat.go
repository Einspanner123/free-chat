package interfaces

import (
	"context"
	"log"
	"time"

	chatpb "free-chat/pkg/proto/chat"
	"free-chat/services/chat-service/internal/application"
	"free-chat/services/chat-service/internal/domain"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type ChatHandler struct {
	chatpb.UnimplementedChatServiceServer
	app *application.ChatService
	llm *LLMClient
}

func NewChatHandler(app *application.ChatService, llm *LLMClient) *ChatHandler {
	return &ChatHandler{
		app: app,
		llm: llm,
	}
}

func (h *ChatHandler) StreamChat(req *chatpb.ChatRequest, stream chatpb.ChatService_StreamChatServer) error {
	ctx := stream.Context()

	// 1. Ensure Session
	sessionID, err := h.app.EnsureSession(ctx, req.UserId, req.SessionId, req.Message)
	if err != nil {
		return status.Errorf(codes.Internal, "ensure session failed: %v", err)
	}

	// 2. Save User Message
	if err := h.app.SaveMessage(ctx, sessionID, req.UserId, domain.RoleUser, req.Message); err != nil {
		log.Printf("[WARN] save user message failed: %v", err)
		return status.Errorf(codes.Internal, "save message failed: %v", err)
	}

	// 3. Call LLM Service
	// Get full context including the message just saved
	contextStr, err := h.app.GetContext(ctx, sessionID)
	if err != nil {
		log.Printf("[WARN] get context failed: %v", err)
	}

	// Select Best Model Instance (Atomic Select & Increment)
	targetAddr, err := h.app.SelectBestModel(ctx, req.ModelName)
	if err != nil {
		return status.Errorf(codes.Unavailable, "select model instance failed: %v", err)
	}

	defer func() {
		// Use a new context for cleanup to ensure it runs
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err = h.app.DecrementModelLoad(ctx, req.ModelName, targetAddr); err != nil {
			log.Printf("[WARN] failed to decrement model load: %v", err)
		}
	}()

	inferenceReq := &domain.InferenceRequest{
		SessionID: sessionID,
		UserID:    req.UserId,
		Request:   req.Message,
		Model:     targetAddr,
	}

	if contextStr != "" {
		inferenceReq.Request = contextStr
	}

	tokenChan, err := h.llm.GetGeneratedToken(ctx, inferenceReq)
	if err != nil {
		return status.Errorf(codes.Internal, "call llm failed: %v", err)
	}

	// 4. Stream Response & Aggregate
	var fullResponse string
	for token := range tokenChan {
		if token.Error != "" {
			return status.Errorf(codes.Internal, "llm stream error: %v", token.Error)
		}

		// Send to gRPC stream
		if err := stream.Send(&chatpb.ChatResponse{
			SessionId:       sessionID,
			Content:         token.Content,
			GeneratedTokens: token.Count,
			IsFinished:      token.IsLast,
		}); err != nil {
			return err
		}

		fullResponse += token.Content
	}

	// 5. Save Assistant Message
	if fullResponse != "" {
		// Use a detached context for async save to ensure it completes even if stream ends
		saveCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := h.app.SaveMessage(saveCtx, sessionID, req.UserId, domain.RoleAssistant, fullResponse); err != nil {
			log.Printf("[ERROR] save assistant message failed: %v", err)
		}
	}

	return nil
}

func (h *ChatHandler) CreateSession(ctx context.Context, req *chatpb.CreateSessionRequest) (*chatpb.CreateSessionResponse, error) {
	session, err := h.app.CreateSession(ctx, req.UserId, "New Chat")
	if err != nil {
		return nil, status.Errorf(codes.Internal, "create session failed: %v", err)
	}

	return &chatpb.CreateSessionResponse{
		Success:   true,
		SessionId: session.ID,
		Message:   "Session created successfully",
	}, nil
}

func (h *ChatHandler) GetChatHistory(ctx context.Context, req *chatpb.HistoryRequest) (*chatpb.HistoryResponse, error) {
	messages, err := h.app.GetHistory(ctx, req.SessionId, int(req.Limit), int(req.Offset))
	if err != nil {
		return nil, status.Errorf(codes.Internal, "get history failed: %v", err)
	}

	var pbMessages []*chatpb.ChatMessage
	for _, msg := range messages {
		pbMessages = append(pbMessages, &chatpb.ChatMessage{
			Role:      msg.Role.String(),
			Content:   msg.Content,
			Timestamp: msg.CreatedAt.Unix(),
		})
	}

	return &chatpb.HistoryResponse{
		Messages: pbMessages,
		Total:    int32(len(pbMessages)),
	}, nil
}

func (h *ChatHandler) DeleteSession(ctx context.Context, req *chatpb.DeleteSessionRequest) (*chatpb.DeleteSessionResponse, error) {
	if err := h.app.DeleteSession(ctx, req.SessionId, req.UserId); err != nil {
		return nil, status.Errorf(codes.Internal, "delete session failed: %v", err)
	}

	return &chatpb.DeleteSessionResponse{
		Success: true,
		Message: "Session deleted successfully",
	}, nil
}
