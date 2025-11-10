package handler

import (
	"net/http"

	"free-chat/shared/proto/auth"
	"free-chat/shared/registry"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func LoginHandler(consulRegistry *registry.ConsulRegistry, authServiceName string) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			Username string `json:"username" binding:"required"`
			Password string `json:"password" binding:"required"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "参数错误"})
			return
		}

		instances, err := consulRegistry.DiscoverService(authServiceName)
		if err != nil || len(instances) == 0 {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "认证服务不可用"})
			return
		}

		conn, err := grpc.NewClient(instances[0].GetURL(), grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "连接认证服务失败"})
			return
		}
		defer conn.Close()

		client := auth.NewAuthServiceClient(conn)
		resp, err := client.Login(c.Request.Context(), &auth.LoginRequest{
			Username: req.Username,
			Password: req.Password,
		})
		if err != nil || resp.Error != "" {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "登录失败"})
			return
		}
		c.JSON(http.StatusOK, gin.H{
			"jwt_token": resp.JwtToken,
			"user_id":   resp.UserId,
		})
	}
}

// 可选：注册代理（需要 proto 暴露 Register RPC）
// func RegisterHandler(consulRegistry *registry.ConsulRegistry, authServiceName string) gin.HandlerFunc {
// 	return func(c *gin.Context) {
// 		var req struct {
// 			Username string `json:"username" binding:"required"`
// 			Password string `json:"password" binding:"required"`
// 		}
// 		if err := c.ShouldBindJSON(&req); err != nil {
// 			c.JSON(http.StatusBadRequest, gin.H{"error": "参数错误"})
// 			return
// 		}

// 		instances, err := consulRegistry.DiscoverService(authServiceName)
// 		if err != nil || len(instances) == 0 {
// 			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "认证服务不可用"})
// 			return
// 		}

// 		conn, err := grpc.NewClient(instances[0].GetURL(), grpc.WithTransportCredentials(insecure.NewCredentials()))
// 		if err != nil {
// 			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "连接认证服务失败"})
// 			return
// 		}
// 		defer conn.Close()

// 		client := auth.NewAuthServiceClient(conn)
// 		resp, err := client.Register(c.Request.Context(), &auth.RegisterRequest{
// 			Username: req.Username,
// 			Password: req.Password,
// 		})
// 		if err != nil || resp.Error != "" {
// 			c.JSON(http.StatusBadRequest, gin.H{"error": "注册失败"})
// 			return
// 		}
// 		c.JSON(http.StatusOK, gin.H{
// 			"jwt_token": resp.JwtToken,
// 		})
// 	}
// }
