package adapter

import (
	"context"
	"fmt"
	"free-chat/pkg/registry"
	"free-chat/services/chat-service/internal/infrastructure/persistence/cache"
)

type ModelRepositoryAdapter struct {
	cache     *cache.RedisCache
	discovery *registry.ServiceManager
}

func NewModelRepositoryAdapter(cache *cache.RedisCache, discovery *registry.ServiceManager) *ModelRepositoryAdapter {
	return &ModelRepositoryAdapter{
		cache:     cache,
		discovery: discovery,
	}
}

func (a *ModelRepositoryAdapter) SelectAndIncreaseModelLoads(ctx context.Context, modelName string) (string, error) {
	if a.discovery == nil {
		return "", fmt.Errorf("service discovery is not initialized")
	}
	instances, err := a.discovery.DiscoverService(modelName)
	if err != nil {
		return "", fmt.Errorf("failed to discover service %s: %w", modelName, err)
	}
	if len(instances) == 0 {
		return "", fmt.Errorf("no healthy instances found for service %s", modelName)
	}

	// 2. Extract addresses
	addrs := make([]string, 0, len(instances))
	for _, inst := range instances {
		// Combine Address and Port
		addr := fmt.Sprintf("%s:%d", inst.Address, inst.Port)
		addrs = append(addrs, addr)
	}

	// 3. Select best instance using Redis load tracking
	return a.cache.SelectAndRegister(ctx, modelName, addrs)
}

func (a *ModelRepositoryAdapter) DecrementTaskCount(ctx context.Context, modelName, instanceAddr string) error {
	return a.cache.DecrementTaskCount(ctx, modelName, instanceAddr)
}
