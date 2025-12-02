package handler

import (
	chatpb "free-chat/pkg/proto/chat"
	"free-chat/services/chat-service/internal/application"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type ChatHandler struct {
	chatpb.UnimplementedChatServiceServer
	app *application.ChatService
}

func NewChatHandler(app *application.ChatService) *ChatHandler {
	return &ChatHandler{app: app}
}

func (h *ChatHandler) StreamChat(req *chatpb.ChatRequest, stream chatpb.ChatService_StreamChatServer) error {
	tokenChan, errChan, err := h.app.StreamChat(stream.Context(), req.UserId, req.SessionId, req.Message, req.ModelName)
	if err != nil {
		return status.Errorf(codes.Internal, "failed to start chat: %v", err)
	}

	for {
		select {
		case err, ok := <-errChan:
			if !ok {
				continue
			} // Channel closed
			return status.Errorf(codes.Internal, "llm error: %v", err)
		case content, ok := <-tokenChan:
			if !ok {
				return nil // Stream finished
			}
			if err := stream.Send(&chatpb.ChatResponse{
				SessionId: req.SessionId, // 或新生成的ID
				Content:   content,
			}); err != nil {
				return err
			}
		}
	}
}
