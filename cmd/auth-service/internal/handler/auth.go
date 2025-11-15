package handler

import (
	"context"
	authpb "free-chat/shared/proto/auth"
	"log"

	"free-chat/cmd/auth-service/internal/service"
	"free-chat/cmd/auth-service/internal/store"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type AuthHandler struct {
	userRepo   *store.UserRepository
	jwtService *service.JWTService
	authpb.UnimplementedAuthServiceServer
}

func NewAuthHandler(userRepo *store.UserRepository, jwtService *service.JWTService) *AuthHandler {
	return &AuthHandler{
		userRepo:   userRepo,
		jwtService: jwtService,
	}
}

func (h *AuthHandler) Login(ctx context.Context, req *authpb.LoginRequest) (*authpb.LoginResponse, error) {
	log.Printf("Login request for user: %s", req.Username)

	// 获取用户
	user, err := h.userRepo.GetUserByUsername(req.Username)
	if err != nil {
		return &authpb.LoginResponse{
			Success: false,
			Message: "Invalid credentials",
		}, nil
	}

	// 验证密码
	if !user.CheckPassword(req.Password) {
		return &authpb.LoginResponse{
			Success: false,
			Message: "Invalid credentials",
		}, nil
	}

	// 生成JWT令牌
	accessToken, accessExpireAt, err := h.jwtService.GenerateAccessToken(user.ID, user.Username)
	if err != nil {
		log.Printf("Failed to generate access token: %v", err)
		return nil, status.Error(codes.Internal, "Failed to generate access token")
	}

	refreshToken, _, err := h.jwtService.GenerateRefreshToken(user.ID, user.Username)
	if err != nil {
		log.Printf("Failed to generate refresh token: %v", err)
		return nil, status.Error(codes.Internal, "Failed to generate refresh token")
	}

	return &authpb.LoginResponse{
		Success:      true,
		Message:      "Login successful",
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
		ExpiresAt:    accessExpireAt.Unix(),
	}, nil
}

func (h *AuthHandler) Register(ctx context.Context, req *authpb.RegisterRequest) (*authpb.RegisterResponse, error) {
	log.Printf("Register request for user: %s", req.Username)

	// 检查用户名是否已存在
	if _, err := h.userRepo.GetUserByUsername(req.Username); err == nil {
		return &authpb.RegisterResponse{
			Success: false,
			Message: "Username already exists",
		}, nil
	}

	// 检查邮箱是否已存在
	if _, err := h.userRepo.GetUserByEmail(req.Email); err == nil {
		return &authpb.RegisterResponse{
			Success: false,
			Message: "Email already exists",
		}, nil
	}

	// 创建用户

	if _, err := h.userRepo.CreateUser(req.Username, req.Email, req.Password); err != nil {
		log.Printf("Failed to create user: %v", err)
		return nil, status.Error(codes.Internal, "Failed to create user")
	}

	return &authpb.RegisterResponse{
		Success: true,
		Message: "User registered successfully",
	}, nil
}

func (h *AuthHandler) ValidateToken(ctx context.Context, req *authpb.ValidateTokenRequest) (*authpb.ValidateTokenResponse, error) {
	claims, err := h.jwtService.ValidateToken(req.Token)
	if err != nil {
		return &authpb.ValidateTokenResponse{
			Valid: false,
		}, nil
	}

	// 获取用户信息
	_, err = h.userRepo.GetUserByID(claims.UserID)
	if err != nil {
		return &authpb.ValidateTokenResponse{
			Valid: false,
		}, nil
	}

	return &authpb.ValidateTokenResponse{
		Valid: true,
	}, nil
}

func (h *AuthHandler) RefreshToken(ctx context.Context, req *authpb.RefreshTokenRequest) (*authpb.RefreshTokenResponse, error) {
	newToken, expirationRefresh, err := h.jwtService.RefreshToken(req.RefreshToken)
	if err != nil {
		return &authpb.RefreshTokenResponse{
			Success: false,
		}, nil
	}

	return &authpb.RefreshTokenResponse{
		Success:     true,
		AccessToken: newToken,
		ExpiresAt:   expirationRefresh.Unix(),
	}, nil
}
