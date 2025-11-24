package service

import (
	"context"
	llmpb "free-chat/shared/proto/llm_inference"
	"free-chat/shared/registry"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

type LLMClient struct {
	mgr *registry.ServiceManager
}

func NewLLMClient(mgr *registry.ServiceManager) *LLMClient {
	return &LLMClient{
		mgr: mgr,
	}
}

func (c *LLMClient) StreamInference(ctx context.Context, sessionID, message, llmUrl string) (llmpb.InferencerService_StreamInferenceClient, error) {
	// 传递 Trace ID
	ctx = metadata.AppendToOutgoingContext(ctx, "x-trace-id", sessionID)

	req := &llmpb.InferenceRequest{
		SessionId: sessionID,
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
	// go func() {
	// 	defer close(ch)
	// 	for {
	// 		resp, err := stream.Recv()
	// 		if err == io.EOF {
	// 			break
	// 		} else if err != nil {

	// 		}
	// 		ch <- resp
	// 	}
	// 	for i, chunk := range chunks {
	// 		resp := &llmpb.InferenceResponse{
	// 			Chunk:           chunk,
	// 			IsFinished:      i == len(chunks)-1,
	// 			GeneratedTokens: int32(len(chunk)),
	// 		}
	// 		ch <- resp
	// 	}
	// }()

	return stream, nil
}
