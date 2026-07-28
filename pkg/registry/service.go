package registry

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
)

type ServiceConfig struct {
	ID          string
	Name        string
	Tags        []string
	Address     string
	Port        int
	HealthCheck *HealthCheck
}

// 服务实例信息
type ServiceInstance struct {
	ID      string
	Name    string
	Address string
	Port    int
	Tags    []string
}

// 获取服务URL
func (s *ServiceInstance) GetEndpoint() string {
	return fmt.Sprintf("%s:%d", s.Address, s.Port)
}

// 服务管理器
type ServiceManager struct {
	registry      *ConsulRegistry
	serviceConfig *ServiceConfig
	stopChan      chan os.Signal
}

// 创建服务管理器
func NewServiceManager(consulConfig *ConsulConfig, serviceConfig *ServiceConfig) (*ServiceManager, error) {
	// 创建Consul注册器
	consulRegistry, err := NewConsulRegistry(consulConfig)
	if err != nil {
		return nil, err
	}

	// 设置信号处理
	stopChan := make(chan os.Signal, 1)
	signal.Notify(stopChan, syscall.SIGINT, syscall.SIGTERM)

	return &ServiceManager{
		registry:      consulRegistry,
		serviceConfig: serviceConfig,
		stopChan:      stopChan,
	}, nil
}

// 启动服务
func (sm *ServiceManager) Start() error {
	// 注册服务
	if err := sm.registry.RegisterService(sm.serviceConfig); err != nil {
		return fmt.Errorf("service registration failed: %v", err)
	}
	serviceName := sm.serviceConfig.Name

	log.Printf("%s started", serviceName)
	return nil
}

// 停止服务
func (sm *ServiceManager) Stop() {
	// 注销服务
	if err := sm.registry.DeregisterService(sm.serviceConfig.ID); err != nil {
		log.Printf("service deregistration failed: %v", err)
	}
}

func (sm *ServiceManager) DiscoverService(serviceName string) ([]*ServiceInstance, error) {
	return sm.registry.DiscoverService(serviceName)
}
