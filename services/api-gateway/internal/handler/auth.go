package handler

import (
	"fmt"
	"math/rand"
	"net/http"
	"sync"

	authpb "free-chat/pkg/proto/auth"
	"free-chat/pkg/registry"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

type AuthHandler struct {
	mgr         *registry.ServiceManager
	authService string
	mu          sync.RWMutex
	conns       map[string]*grpc.ClientConn
}

func (h *AuthHandler) Close() {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, conn := range h.conns {
		conn.Close()
	}
	h.conns = make(map[string]*grpc.ClientConn)
}

func NewAuthHandler(mgr *registry.ServiceManager, authService string) *AuthHandler {
	return &AuthHandler{
		mgr:         mgr,
		authService: authService,
		conns:       make(map[string]*grpc.ClientConn),
	}
}

func (h *AuthHandler) getConn(target string) (*grpc.ClientConn, error) {
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

func (h *AuthHandler) getGRPCConnection() (*grpc.ClientConn, error) {
	instances, err := h.mgr.DiscoverService(h.authService)
	if err != nil {
		return nil, err
	}
	if len(instances) == 0 {
		return nil, status.Error(codes.Unavailable, fmt.Sprintf("no auth service instances found for %s", h.authService))
	}

	// Random load balancing
	select_inst := instances[rand.Intn(len(instances))]

	return h.getConn(select_inst.GetEndpoint())
}

func (h *AuthHandler) Login(c *gin.Context) {
	var req struct {
		UserName string `json:"user_name" binding:"required"`
		Password string `json:"password" binding:"required"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效请求参数：" + err.Error()})
		return
	}
	// 连接到认证服务
	conn, err := h.getGRPCConnection()
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "连接认证服务失败：" + err.Error()})
		return
	}
	// Do not close pooled connection

	client := authpb.NewAuthServiceClient(conn)
	resp, err := client.Login(c.Request.Context(), &authpb.LoginRequest{
		Username: req.UserName,
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
		UserName string `json:"user_name" binding:"required"`
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

	client := authpb.NewAuthServiceClient(conn)
	resp, err := client.Register(c.Request.Context(), &authpb.RegisterRequest{
		Username: req.UserName,
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

	client := authpb.NewAuthServiceClient(conn)
	resp, err := client.RefreshToken(c.Request.Context(), &authpb.RefreshTokenRequest{
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
