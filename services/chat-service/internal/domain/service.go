package domain

import "context"

type InferenceService interface {
	StreamInference(ctx context.Context, req *InferenceRequest) (<-chan *GeneratedToken, error)
}

type ModelBalanceService interface {
	SelectAndIncreaseModelLoads(ctx context.Context, modelName string) (string, error)
	DecrementTaskCount(ctx context.Context, modelName, instanceAddr string) error
}

// ContextOptimizer builds optimized contexts under a token budget.
// Implemented by the remote context-engine client (Python service).
type ContextOptimizer interface {
	BuildContext(ctx context.Context, text, query, strategy string, budget int) (string, error)
}
