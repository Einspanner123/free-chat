package service

import (
	"fmt"
	"free-chat/shared/registry"
	"log"
	"os"
	"os/signal"
	"syscall"
)

// 服务管理器
type ServiceManager struct {
	registry      *registry.ConsulRegistry
	serviceConfig *registry.ServiceConfig
	stopChan      chan os.Signal
}

// 创建服务管理器
func NewServiceManager(consulConfig *registry.ConsulConfig, serviceConfig *registry.ServiceConfig) (*ServiceManager, error) {
	// 创建Consul注册器
	consulRegistry, err := registry.NewConsulRegistry(consulConfig)
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
		return fmt.Errorf("服务注册失败: %v", err)
	}

	// 启动优雅关闭监听
	go sm.gracefulShutdown()

	log.Println("🎯 服务管理器启动成功")
	return nil
}

// 优雅关闭
func (sm *ServiceManager) gracefulShutdown() {
	sm.WaitForShutdown()
	log.Println("🛑 接收到关闭信号，开始优雅关闭...")

	// 注销服务
	if err := sm.registry.DeregisterService(sm.serviceConfig.ID); err != nil {
		log.Printf("❌ 服务注销失败: %v", err)
	}

	log.Println("✅ 服务已优雅关闭")
	os.Exit(0)
}

// 发现服务
func (sm *ServiceManager) DiscoverService(serviceName string) ([]*registry.ServiceInstance, error) {
	return sm.registry.DiscoverService(serviceName)
}

// 等待关闭信号
func (sm *ServiceManager) WaitForShutdown() {
	<-sm.stopChan
}
