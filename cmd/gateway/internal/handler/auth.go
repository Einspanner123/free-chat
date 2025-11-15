package handler

import (
	"context"
	"fmt"
	"net/http"

	authpb "free-chat/shared/proto/auth"
	"free-chat/shared/registry"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type AuthHandler struct {
	mgr         *registry.ServiceManager
	authService string
}

func NewAuthHandler(mgr *registry.ServiceManager, authService string) *AuthHandler {
	return &AuthHandler{
		mgr:         mgr,
		authService: authService,
	}
}

func (h *AuthHandler) getGRPCConnection() (*grpc.ClientConn, error) {
	instances, err := h.mgr.DiscoverService(h.authService)
	if err != nil {
		return nil, err
	}
	if len(instances) == 0 {
		return nil, fmt.Errorf("没有可用的`%s`服务", h.authService)
	}
	select_inst := instances[0] // 进行简单负载均衡选择

	return grpc.NewClient(select_inst.GetURL(), grpc.WithTransportCredentials(insecure.NewCredentials()))

}

func (h *AuthHandler) Login(c *gin.Context) {
	var req struct {
		Username string `json:"username" binding:"required"`
		Password string `json:"password" binding:"required"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效请求参数：" + err.Error()})
		return
	}
	// 连接到认证服务
	conn, err := h.getGRPCConnection()
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "连接聊天服务失败：" + err.Error()})
		return
	}
	defer conn.Close()

	client := authpb.NewAuthServiceClient(conn)
	resp, err := client.Login(context.Background(), &authpb.LoginRequest{
		Username: req.Username,
		Password: req.Password,
	})
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "未授权: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"access_token":  resp.AccessToken,
		"refresh_token": resp.RefreshToken,
	})
}

func (h *AuthHandler) Register(c *gin.Context) {
	var req struct {
		Username string `json:"username" binding:"required"`
		Email    string `json:"email" binding:"required,email"`
		Password string `json:"password" binding:"required,min=6"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 连接到认证服务
	conn, err := h.getGRPCConnection()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Auth service unavailable: " + err.Error()})
		return
	}
	defer conn.Close()

	client := authpb.NewAuthServiceClient(conn)
	resp, err := client.Register(context.Background(), &authpb.RegisterRequest{
		Username: req.Username,
		Email:    req.Email,
		Password: req.Password,
	})
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Registration failed: " + err.Error()})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"success": resp.Success,
		"message": resp.Message,
	})
}

func (h *AuthHandler) RefreshToken(c *gin.Context) {
	var req struct {
		RefreshToken string `json:"refresh_token" binding:"required"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 连接到认证服务
	conn, err := h.getGRPCConnection()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Auth service unavailable"})
		return
	}
	defer conn.Close()

	client := authpb.NewAuthServiceClient(conn)
	resp, err := client.RefreshToken(context.Background(), &authpb.RefreshTokenRequest{
		RefreshToken: req.RefreshToken,
	})

	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Token refresh failed"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success":      resp.Success,
		"access_token": resp.AccessToken,
		"expires_at":   resp.ExpiresAt,
	})
}
