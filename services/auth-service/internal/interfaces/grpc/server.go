package grpc

import (
	"context"
	"fmt"
	authpb "free-chat/pkg/proto/auth"
	"log"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
)

type AuthServer struct {
	registryURL string
	serviceName string
	authHandler *AuthHandler
}

func NewAuthServer(registryURL, serviceName string, authHandler *AuthHandler) *AuthServer {
	return &AuthServer{
		registryURL: registryURL,
		serviceName: serviceName,
		authHandler: authHandler,
	}
}

func (s *AuthServer) Register() error {
	grpcServer := grpc.NewServer()
	authpb.RegisterAuthServiceServer(grpcServer, s.authHandler)
	// 注册gRPC健康检查服务
	healthServer := health.NewServer()
	healthServer.SetServingStatus(s.serviceName, grpc_health_v1.HealthCheckResponse_SERVING)
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)

	// 进行连接性测试，确保服务已正确注册并可访问
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 创建健康检查客户端进行测试
	conn, err := grpc.NewClient(s.registryURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("无法连接到gRPC服务: %v", err)
		return err
	}
	defer conn.Close()

	// 执行健康检查
	healthClient := grpc_health_v1.NewHealthClient(conn)
	healthResp, err := healthClient.Check(ctx, &grpc_health_v1.HealthCheckRequest{
		Service: s.serviceName,
	})
	if err != nil {
		log.Printf("健康检查失败: %v", err)
		return err
	}

	if healthResp.Status != grpc_health_v1.HealthCheckResponse_SERVING {
		log.Printf("服务状态异常: %v", healthResp.Status)
		return fmt.Errorf("服务未处于SERVING状态: %v", healthResp.Status)
	}

	log.Println("服务连接性测试成功")
	return nil
}
