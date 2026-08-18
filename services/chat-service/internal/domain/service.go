package domain

import "context"

type InferenceService interface {
	StreamInference(ctx context.Context, req *InferenceRequest) (<-chan *GeneratedToken, error)
}

type ModelBalanceService interface {
	SelectAndIncreaseModelLoads(ctx context.Context, modelName string) (string, error)
	DecrementTaskCount(ctx context.Context, modelName, instanceAddr string) error
}

// ContextOptimizationResult carries an optimized context plus routing metadata.
type ContextOptimizationResult struct {
	Context          string
	Strategy         string
	Tokens           int
	CompressionRatio float64
}

// ContextOptimizer builds optimized contexts under a token budget.
// Implemented by the remote context-engine client (Python service).
type ContextOptimizer interface {
	BuildContext(ctx context.Context, text, query, strategy string, budget int) (*ContextOptimizationResult, error)
}
