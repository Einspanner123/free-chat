package domain

import "context"

type InferenceService interface {
	StreamInference(ctx context.Context, req *InferenceRequest) (<-chan *GeneratedToken, error)
}

type ModelBalanceService interface {
	SelectAndIncreaseModelLoads(ctx context.Context, modelName string) (string, error)
	DecrementTaskCount(ctx context.Context, modelName, instanceAddr string) error
}
