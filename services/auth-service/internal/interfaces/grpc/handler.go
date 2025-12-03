package grpc

import (
	"context"
	authpb "free-chat/pkg/proto/auth"
	"free-chat/services/auth-service/internal/application"
)

type AuthHandler struct {
	authpb.UnimplementedAuthServiceServer
	authSvc *application.AuthService
}

func NewAuthHandler(svc *application.AuthService) *AuthHandler {
	return &AuthHandler{authSvc: svc}
}

func (s *AuthHandler) Register(ctx context.Context, req *authpb.RegisterRequest) (*authpb.RegisterResponse, error) {
	dtoReq := ToRegisterDTO(req)
	dtoResp, err := s.authSvc.Register(dtoReq)
	if err != nil {
		// 可以在这里转换特定的领域错误到 gRPC 状态码
		return &authpb.RegisterResponse{
			Success: false,
			Message: err.Error(),
		}, nil
	}
	return ToRegisterResponseRPC(dtoResp), nil
}

func (s *AuthHandler) Login(ctx context.Context, req *authpb.LoginRequest) (*authpb.LoginResponse, error) {
	dtoReq := ToLoginDTO(req)
	dtoResp, err := s.authSvc.Login(dtoReq)
	if err != nil {
		return &authpb.LoginResponse{
			Success: false,
			Message: err.Error(),
		}, nil
	}
	return ToLoginResponseRPC(dtoResp), nil
}

func (s *AuthHandler) ValidateToken(ctx context.Context, req *authpb.ValidateTokenRequest) (*authpb.ValidateTokenResponse, error) {
	valid, err := s.authSvc.ValidateToken(req.Token)
	if err != nil {
		return &authpb.ValidateTokenResponse{
			Valid: false,
		}, nil
	}
	return &authpb.ValidateTokenResponse{
		Valid: valid,
		// ExpiresAt: 0, // TokenService 目前不返回过期时间
	}, nil
}

func (s *AuthHandler) RefreshToken(ctx context.Context, req *authpb.RefreshTokenRequest) (*authpb.RefreshTokenResponse, error) {
	dtoResp, err := s.authSvc.RefreshToken(req.RefreshToken)
	if err != nil {
		return &authpb.RefreshTokenResponse{
			Success: false,
		}, nil
	}
	return &authpb.RefreshTokenResponse{
		Success:     true,
		AccessToken: dtoResp.AccessToken,
		ExpiresAt:   dtoResp.ExpiresAt,
	}, nil
}
