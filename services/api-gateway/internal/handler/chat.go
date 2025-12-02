package handler

import (
	"context"
	"log"
	"net/http"

	"free-chat/infra/registry"
	chatpb "free-chat/pkg/proto/chat"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

type ChatHandler struct {
	mgr         *registry.ServiceManager
	chatService string
	llmService  string
}

func NewChatHandler(mgr *registry.ServiceManager, chatService, llmService string) *ChatHandler {
	return &ChatHandler{
		mgr:         mgr,
		chatService: chatService,
		llmService:  llmService,
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
	resp, err := client.CreateSession(
		context.Background(),
		&chatpb.CreateSessionRequest{
			UserId: userID,
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

func (h *ChatHandler) StreamChat(c *gin.Context) {
	var req struct {
		Message   string `json:"message" binding:"required"`
		UserId    string `json:"user_id" binding:"required"`
		SessionId string `json:"session_id" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		log.Printf("req binding error: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 连接到聊天服务
	conn, err := h.getGRPCConnection()
	if err != nil {
		log.Printf("Chat service unavailable: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Chat service unavailable"})
		return
	}
	defer conn.Close()
	client := chatpb.NewChatServiceClient(conn)

	// 创建流式聊天请求
	stream, err := client.StreamChat(context.Background(), &chatpb.ChatRequest{
		SessionId: req.SessionId,
		UserId:    req.UserId,
		Message:   req.Message,
		ModelName: h.llmService,
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
	flushCounter := 0
	for {
		resp, err := stream.Recv()
		if err != nil {
			// 处理stream错误
			if grpcStatus, ok := status.FromError(err); ok {
				c.SSEvent("error", gin.H{
					"message": grpcStatus.Message(),
					"code":    grpcStatus.Code(),
				})
			} else {
				c.SSEvent("error", gin.H{
					"message": "Unknown error occurred",
				})
			}
			c.Writer.Flush()
			break
		}
		// 处理后端错误
		if resp.Error != "" {
			c.SSEvent("error", gin.H{
				"message": resp.Error,
			})
			c.Writer.Flush()
			break
		}
		c.SSEvent("message", gin.H{
			"content":   resp.Content,
			"finished":  resp.IsFinished,
			"sessionId": resp.SessionId,
		})
		flushCounter++
		if resp.IsFinished || flushCounter >= 5 { // 硬编码，后续优化
			c.Writer.Flush()
			flushCounter = 0
		}

		if resp.IsFinished {
			break
		}
	}
}
