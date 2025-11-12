package handler

import (
	"encoding/json"
	"free-chat/shared/proto/chat"
	"free-chat/shared/registry"
	"net/http"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func ChatHandler(service *registry.ServiceManager, chatServiceName string) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			SessionID string `json:"session_id"`
			Message   string `json:"message" binding:"required"`
			Model     string `json:"model" binding:"required"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "无效请求参数：" + err.Error()})
			return
		}

		userID, exists := c.Get("user_id")
		if !exists {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "未认证"})
			return
		}

		instances, err := service.DiscoverService(chatServiceName)
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "聊天服务暂时不可用：" + err.Error()})
			return
		}

		conn, err := grpc.NewClient(instances[0].GetURL(), grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "连接聊天服务失败：" + err.Error()})
			return
		}
		defer conn.Close()
		chatClient := chat.NewChatServiceClient(conn)

		stream, err := chatClient.StreamChat(c.Request.Context(), &chat.ChatRequest{
			UserId:    userID.(string),
			SessionId: req.SessionID,
			Message:   req.Message,
			Model:     req.Model,
		})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "启动聊天失败：" + err.Error()})
			return
		}

		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")

		for {
			resp, err := stream.Recv()
			if err != nil {
				break
			}
			payload := map[string]interface{}{
				"session_id":  resp.SessionId,
				"token":       resp.Token,
				"is_finished": resp.IsFinished,
			}
			b, _ := json.Marshal(payload)
			c.Writer.WriteString("data: " + string(b) + "\n\n")
			c.Writer.Flush()
			if resp.IsFinished {
				break
			}
		}
	}
}

// func ChatHandler(registry *registry.ConsulRegistry, chatServiceName string) gin.HandlerFunc {
// 	return func(c *gin.Context) {
// 		var req struct {
// 			SessionID string `json:"session_id"`
// 			Query     string `json:"query" binding:"required"`
// 			Model     string `json:"model" binding:"required"`
// 		}
// 		if err := c.ShouldBindJSON(&req); err != nil {
// 			c.JSON(http.StatusBadRequest, gin.H{"error": "无效请求参数：" + err.Error()})
// 			return
// 		}

// 		userID, exists := c.Get("user_id")
// 		if !exists {
// 			c.JSON(http.StatusUnauthorized, gin.H{"error": "未认证"})
// 			return
// 		}

// 		chatAddr, err := registry.DiscoverService(chatServiceName)
// 		if err != nil {
// 			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "聊天服务暂时不可用：" + err.Error()})
// 			return
// 		}

// 		conn, err := grpc.NewClient(chatAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
// 		if err != nil {
// 			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "连接聊天服务失败：" + err.Error()})
// 			return
// 		}
// 		defer conn.Close()
// 		chatClient := chat.NewChatServiceClient(conn)

// 		stream, err := chatClient.StreamChat(context.Background(), &chat.ChatRequest{
// 			UserId:    userID.(string),
// 			SessionId: req.SessionID,
// 			Query:     req.Query,
// 			Model:     req.Model,
// 		})
// 		if err != nil {
// 			c.JSON(http.StatusInternalServerError, gin.H{"error": "启动聊天失败：" + err.Error()})
// 			return
// 		}

// 		c.Header("Content-Type", "text/event-stream")
// 		c.Header("Cache-Control", "no-cache")
// 		c.Header("Connection", "keep-alive")

// 		// 7. 接收gRPC流式响应，转发给前端
// 		for {
// 			resp, err := stream.Recv()
// 			if err != nil {
// 				break // 流结束
// 			}
// 			// 以JSON格式返回（SSE规范：data: {json}\n\n）
// 			c.JSON(http.StatusOK, gin.H{
// 				"session_id":  resp.SessionId,
// 				"token":       resp.Token,
// 				"is_finished": resp.IsFinished,
// 			})
// 			c.Writer.Flush() // 立即推送数据

// 			if resp.IsFinished {
// 				break // 生成完毕，结束流
// 			}
// 		}
// 	}
// }
