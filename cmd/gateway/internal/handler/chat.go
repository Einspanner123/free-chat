package handler

import (
	"context"
	"net/http"

	chatpb "free-chat/shared/proto/chat"
	"free-chat/shared/registry"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type ChatHandler struct {
	mgr         *registry.ServiceManager
	chatService string
}

func NewChatHandler(mgr *registry.ServiceManager, chatService string) *ChatHandler {
	return &ChatHandler{
		mgr:         mgr,
		chatService: chatService,
	}
}

func (h *ChatHandler) getGRPCConnection() (*grpc.ClientConn, error) {
	instances, err := h.mgr.DiscoverService(h.chatService)
	if err != nil {
		return nil, err
	}
	select_inst := instances[0] // 进行简单负载均衡选择

	return grpc.NewClient(select_inst.GetURL(), grpc.WithTransportCredentials(insecure.NewCredentials()))

}

func (h *ChatHandler) CreateSession(c *gin.Context) {
	userID := c.GetString("user_id")
	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
		return
	}

	var req struct {
		Title string `json:"title"`
	}
	c.ShouldBindJSON(&req)

	// 连接到聊天服务
	conn, err := h.getGRPCConnection()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Chat service unavailable"})
		return
	}
	defer conn.Close()

	client := chatpb.NewChatServiceClient(conn)
	resp, err := client.CreateSession(context.Background(), &chatpb.CreateSessionRequest{
		UserId: userID,
		Title:  req.Title,
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create session"})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"success":    resp.Success,
		"session_id": resp.SessionId,
		"message":    resp.Message,
	})
}

func (h *ChatHandler) GetHistory(c *gin.Context) {
	sessionID := c.Param("sessionId")
	userID := c.GetString("user_id")

	// 连接到聊天服务
	conn, err := h.getGRPCConnection()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Chat service unavailable"})
		return
	}
	defer conn.Close()

	client := chatpb.NewChatServiceClient(conn)
	resp, err := client.GetChatHistory(context.Background(), &chatpb.HistoryRequest{
		SessionId: sessionID,
		UserId:    userID,
	})

	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Session not found"})
		return
	}

	messages := make([]gin.H, len(resp.Messages))
	for i, msg := range resp.Messages {
		messages[i] = gin.H{
			"id":        msg.Id,
			"role":      msg.Role,
			"content":   msg.Content,
			"timestamp": msg.Timestamp,
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"messages": messages,
		"total":    resp.Total,
	})
}

func (h *ChatHandler) DeleteSession(c *gin.Context) {
	sessionID := c.Param("sessionId")
	userID := c.GetString("user_id")

	// 连接到聊天服务
	conn, err := h.getGRPCConnection()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Chat service unavailable"})
		return
	}
	defer conn.Close()

	client := chatpb.NewChatServiceClient(conn)
	resp, err := client.DeleteSession(context.Background(), &chatpb.DeleteSessionRequest{
		SessionId: sessionID,
		UserId:    userID,
	})

	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Session not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": resp.Success,
		"message": resp.Message,
	})
}

func (h *ChatHandler) SendMessage(c *gin.Context) {
	sessionID := c.Param("sessionId")
	userID := c.GetString("user_id")

	var req struct {
		Content   string `json:"content" binding:"required"`
		ModelName string `json:"model_name"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 连接到聊天服务
	conn, err := h.getGRPCConnection()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Chat service unavailable"})
		return
	}
	defer conn.Close()

	client := chatpb.NewChatServiceClient(conn)

	// 创建流式聊天请求
	stream, err := client.StreamChat(context.Background(), &chatpb.ChatRequest{
		SessionId: sessionID,
		UserId:    userID,
		Message:   req.Content,
		Model:     req.ModelName,
		Config: &chatpb.ChatConfig{
			Temperature: 0.7,
			MaxTokens:   2048,
		},
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to send message"})
		return
	}

	// 设置SSE头
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")

	// 流式响应
	for {
		resp, err := stream.Recv()
		if err != nil {
			break
		}

		c.SSEvent("message", gin.H{
			"token":     resp.Token,
			"finished":  resp.IsFinished,
			"sessionId": resp.SessionId,
		})
		c.Writer.Flush()

		if resp.IsFinished {
			break
		}
	}
}

func (h *ChatHandler) StreamChat(c *gin.Context) {
	// 这个方法与SendMessage类似，但专门用于WebSocket连接
	h.SendMessage(c)
}
