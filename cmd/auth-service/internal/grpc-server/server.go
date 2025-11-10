package server

import (
	"context"
	"free-chat/cmd/auth-service/internal/service"
	"free-chat/shared/proto/auth"
)

type AuthGRPCServer struct {
	auth.UnimplementedAuthServiceServer
	Svc *service.AuthService
}

func (s *AuthGRPCServer) Login(ctx context.Context, req *auth.LoginRequest) (*auth.LoginResponse, error) {
	token, userID, err := s.Svc.Login(req.Username, req.Password)
	if err != nil {
		return &auth.LoginResponse{Error: err.Error()}, nil
	}
	return &auth.LoginResponse{JwtToken: token, UserId: userID}, nil
}

func (s *AuthGRPCServer) VerifyToken(ctx context.Context, req *auth.VerifyTokenRequest) (*auth.VerifyTokenResponse, error) {
	userID, err := s.Svc.VerifyToken(req.JwtToken)
	if err != nil {
		return &auth.VerifyTokenResponse{Valid: false, Error: err.Error()}, nil
	}
	return &auth.VerifyTokenResponse{Valid: true, UserId: userID}, nil
}

// 可选：注册
// func (s *AuthGRPCServer) Register(ctx context.Context, req *auth.RegisterRequest) (*auth.RegisterResponse, error) {
// 	token, err := s.Svc.Register(req.Username, req.Password)
// 	if err != nil {
// 		return &auth.RegisterResponse{Error: err.Error()}, nil
// 	}
// 	return &auth.RegisterResponse{JwtToken: token}, nil
// }
