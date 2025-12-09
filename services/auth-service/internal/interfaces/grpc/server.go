package grpc

import (
	"fmt"
	authpb "free-chat/pkg/proto/auth"
	"net"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
)

type AuthServer struct {
	registryEndpoint string
	serviceName      string
	authHandler      *AuthHandler
	server           *grpc.Server
}

func NewAuthServer(registryEndpoint, serviceName string, authHandler *AuthHandler) *AuthServer {
	return &AuthServer{
		registryEndpoint: registryEndpoint,
		serviceName:      serviceName,
		authHandler:      authHandler,
	}
}

func (s *AuthServer) Serve(servePort int) error {
	s.server = grpc.NewServer()
	authpb.RegisterAuthServiceServer(s.server, s.authHandler)
	// 注册gRPC健康检查服务
	healthServer := health.NewServer()
	healthServer.SetServingStatus(s.serviceName, grpc_health_v1.HealthCheckResponse_SERVING)
	grpc_health_v1.RegisterHealthServer(s.server, healthServer)

	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", servePort))
	if err != nil {
		return err
	}
	return s.server.Serve(listener)
}

func (s *AuthServer) Stop() {
	if s.server != nil {
		s.server.GracefulStop()
	}
}
