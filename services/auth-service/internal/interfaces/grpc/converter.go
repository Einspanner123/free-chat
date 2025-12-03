package grpc

import (
	authpb "free-chat/pkg/proto/auth"
	"free-chat/services/auth-service/internal/application/dto"
)

// ToLoginDTO converts gRPC LoginRequest to application layer LoginReq DTO
func ToLoginDTO(req *authpb.LoginRequest) *dto.LoginReq {
	if req == nil {
		return nil
	}
	return &dto.LoginReq{
		Username: req.UserName,
		Password: req.Password,
	}
}

// ToLoginRPC converts application layer LoginReq DTO to gRPC LoginRequest
// Note: This is primarily for testing or internal client usage
func ToLoginRPC(req *dto.LoginReq) *authpb.LoginRequest {
	if req == nil {
		return nil
	}
	return &authpb.LoginRequest{
		UserName: req.Username,
		Password: req.Password,
	}
}

// ToLoginResponseRPC converts application layer LoginResp DTO to gRPC LoginResponse
func ToLoginResponseRPC(resp *dto.LoginResp) *authpb.LoginResponse {
	if resp == nil {
		return &authpb.LoginResponse{
			Success: false,
			Message: "Login failed",
		}
	}
	return &authpb.LoginResponse{
		Success:      true,
		Message:      "Login successful",
		AccessToken:  resp.AccessToken,
		RefreshToken: resp.RefreshToken,
		ExpiresAt:    resp.ExpiresAt,
	}
}

// ToRegisterDTO converts gRPC RegisterRequest to application layer RegisterReq DTO
func ToRegisterDTO(req *authpb.RegisterRequest) *dto.RegisterReq {
	if req == nil {
		return nil
	}
	return &dto.RegisterReq{
		Username: req.UserName,
		Password: req.Password,
		Email:    req.Email,
	}
}

// ToRegisterResponseRPC converts application layer RegisterResp DTO to gRPC RegisterResponse
func ToRegisterResponseRPC(resp *dto.RegisterResp) *authpb.RegisterResponse {
	if resp == nil {
		return &authpb.RegisterResponse{
			Success: false,
			Message: "Registration failed",
		}
	}
	return &authpb.RegisterResponse{
		Success: true,
		Message: "Registration successful",
		UserId:  resp.UserID,
	}
}
