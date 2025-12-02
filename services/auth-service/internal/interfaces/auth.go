package handler

import (
	"context"
	"fmt"
	"log"

	authpb "free-chat/pkg/proto/auth"
	"free-chat/services/auth-service/internal/application"
	"free-chat/services/auth-service/internal/domain"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type AuthHandler struct {
	authService *application.AuthService
	authpb.UnimplementedAuthServiceServer
}

func NewAuthHandler(authService *application.AuthService) *AuthHandler {
	return &AuthHandler{
		authService: authService,
	}
}

func (h *AuthHandler) Login(ctx context.Context, req *authpb.LoginRequest) (*authpb.LoginResponse, error) {
	result, err := h.authService.Login(ctx, req.UserName, req.Password)
	if err != nil {
		if err == domain.ErrUserNotFound || err == domain.ErrInvalidPassword {
			return &authpb.LoginResponse{
				Success: false,
				Message: "Invalid credentials",
			}, nil
		}
		log.Printf("Login failed: %v", err)
		return nil, status.Error(codes.Internal, "Internal server error")
	}

	return &authpb.LoginResponse{
		Success:      true,
		Message:      "Login successful",
		AccessToken:  result.AccessToken,
		RefreshToken: result.RefreshToken,
		ExpiresAt:    result.ExpiresAt.Unix(),
	}, nil
}

func (h *AuthHandler) Register(ctx context.Context, req *authpb.RegisterRequest) (*authpb.RegisterResponse, error) {
	result, err := h.authService.Register(ctx, req.UserName, req.Email, req.Password)
	if err != nil {
		if err == domain.ErrUserAlreadyExists {
			return &authpb.RegisterResponse{
				Success: false,
				Message: "User already exists",
			}, nil
		}
		log.Printf("Register failed: %v", err)
		return &authpb.RegisterResponse{
			Success: false,
			Message: fmt.Sprintf("Register failed: %v", err),
		}, status.Error(codes.Internal, "Internal server error")
	}

	return &authpb.RegisterResponse{
		Success: true,
		UserId:  result.UserID,
		Message: "User registered successfully",
	}, nil
}

func (h *AuthHandler) ValidateToken(ctx context.Context, req *authpb.ValidateTokenRequest) (*authpb.ValidateTokenResponse, error) {
	claims, err := h.authService.ValidateToken(ctx, req.Token)
	if err != nil {
		return &authpb.ValidateTokenResponse{
			Valid: false,
		}, nil
	}

	return &authpb.ValidateTokenResponse{
		Valid:     true,
		ExpiresAt: claims.ExpiresAt.Unix(),
	}, nil
}

func (h *AuthHandler) RefreshToken(ctx context.Context, req *authpb.RefreshTokenRequest) (*authpb.RefreshTokenResponse, error) {
	newToken, expirationRefresh, err := h.authService.RefreshToken(ctx, req.RefreshToken)
	if err != nil {
		return &authpb.RefreshTokenResponse{
			Success: false,
		}, err
	}

	return &authpb.RefreshTokenResponse{
		Success:     true,
		AccessToken: newToken,
		ExpiresAt:   expirationRefresh.Unix(),
	}, nil
}
