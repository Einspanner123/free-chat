package service

import (
	"context"
	"free-chat/infra/registry"
	llmpb "free-chat/pkg/proto/llm_inference"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

type LLMClient struct {
	svcMgr *registry.ServiceManager
}

func NewLLMClient(serviceManager *registry.ServiceManager) *LLMClient {
	return &LLMClient{
		svcMgr: serviceManager,
	}
}

func (c *LLMClient) StreamInference(ctx context.Context, sessionId, message, llmUrl string) (llmpb.InferencerService_StreamInferenceClient, error) {
	// 传递 Trace ID
	ctx = metadata.AppendToOutgoingContext(ctx, "x-trace-id", sessionId)

	req := &llmpb.InferenceRequest{
		SessionId: sessionId,
		Message:   message,
	}
	conn, err := grpc.NewClient(llmUrl, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	clnt := llmpb.NewInferencerServiceClient(conn)

	stream, err := clnt.StreamInference(ctx)
	if err != nil {
		return nil, err
	}
	if err := stream.Send(req); err != nil {
		return nil, err
	}
	return stream, nil
}
