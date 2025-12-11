package handler

import (
	"io"
	"log"
	"math/rand"
	"net/http"
	"strconv"
	"sync"

	chatpb "free-chat/pkg/proto/chat"
	"free-chat/pkg/registry"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

type ChatHandler struct {
	mgr         *registry.ServiceManager
	chatService string
	llmService  string
	mu          sync.RWMutex
	conns       map[string]*grpc.ClientConn
}

func (h *ChatHandler) Close() {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, conn := range h.conns {
		conn.Close()
	}
	h.conns = make(map[string]*grpc.ClientConn)
}

func NewChatHandler(mgr *registry.ServiceManager, chatService, llmService string) *ChatHandler {
	return &ChatHandler{
		mgr:         mgr,
		chatService: chatService,
		llmService:  llmService,
		conns:       make(map[string]*grpc.ClientConn),
	}
}

func (h *ChatHandler) getConn(target string) (*grpc.ClientConn, error) {
	h.mu.RLock()
	conn, ok := h.conns[target]
	h.mu.RUnlock()

	if ok {
		state := conn.GetState()
		if state == connectivity.Shutdown {
			// Connection is shutdown, need to create a new one
		} else {
			return conn, nil
		}
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	// Double check
	if conn, ok = h.conns[target]; ok {
		if conn.GetState() != connectivity.Shutdown {
			return conn, nil
		}
	}

	// Dial new connection
	conn, err := grpc.NewClient(target, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}

	h.conns[target] = conn
	return conn, nil
}

func (h *ChatHandler) getGRPCConnection() (*grpc.ClientConn, error) {
	instances, err := h.mgr.DiscoverService(h.chatService)
	if err != nil {
		return nil, err
	}
	if len(instances) == 0 {
		return nil, status.Error(codes.Unavailable, "no chat service instances found")
	}

	select_inst := instances[rand.Intn(len(instances))]

	return h.getConn(select_inst.GetEndpoint())
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

	client := chatpb.NewChatServiceClient(conn)
	resp, err := client.CreateSession(
		c.Request.Context(),
		&chatpb.CreateSessionRequest{
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

	client := chatpb.NewChatServiceClient(conn)
	resp, err := client.GetChatHistory(c.Request.Context(), &chatpb.HistoryRequest{
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

	client := chatpb.NewChatServiceClient(conn)
	resp, err := client.DeleteSession(c.Request.Context(), &chatpb.DeleteSessionRequest{
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

func (h *ChatHandler) GetSessions(c *gin.Context) {
	userID := c.GetString("user_id")
	limit := c.Query("limit")
	offset := c.Query("offset")
	limit32, err := strconv.ParseInt(limit, 10, 32)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	offset32, err := strconv.ParseInt(offset, 10, 32)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	conn, err := h.getGRPCConnection()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Chat service unavailable"})
		return
	}

	client := chatpb.NewChatServiceClient(conn)
	resp, err := client.GetSessions(c.Request.Context(), &chatpb.GetSessionsRequest{
		UserId: userID,
		Limit:  int32(limit32),
		Offset: int32(offset32),
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get sessions"})
		return
	}

	sessions := make([]gin.H, len(resp.Sessions))
	for i, s := range resp.Sessions {
		sessions[i] = gin.H{
			"session_id": s.SessionId,
			"title":      s.Title,
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"sessions": sessions,
		"total":    resp.Total,
	})
}

func (h *ChatHandler) StreamChat(c *gin.Context) {
	var req struct {
		Message   string `json:"message" binding:"required"`
		SessionId string `json:"session_id" binding:"required"`
		Model     string `json:"model"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		log.Printf("req binding error: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	userID := c.GetString("user_id")
	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
		return
	}

	// 连接到聊天服务
	conn, err := h.getGRPCConnection()
	if err != nil {
		log.Printf("Chat service unavailable: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Chat service unavailable"})
		return
	}

	client := chatpb.NewChatServiceClient(conn)

	model := req.Model
	if model == "" {
		model = h.llmService
	}

	// 创建流式聊天请求
	stream, err := client.StreamChat(c.Request.Context(), &chatpb.ChatRequest{
		SessionId: req.SessionId,
		UserId:    userID,
		Message:   req.Message,
		ModelName: model,
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
		if err == io.EOF {
			c.Writer.Flush()
			break
		}
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
