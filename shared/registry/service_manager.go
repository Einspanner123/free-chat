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
func (s *ServiceInstance) GetURL() string {
	return fmt.Sprintf("http://%s:%d", s.Address, s.Port)
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
		return fmt.Errorf("服务注册失败: %v", err)
	}
	serviceName := sm.serviceConfig.Name

	// 启动优雅关闭监听
	go sm.gracefulShutdown()

	log.Printf("🎯 %s 服务启动成功", serviceName)
	return nil
}

// 优雅关闭
func (sm *ServiceManager) gracefulShutdown() {
	sm.waitForShutdown()
	log.Println("🛑 接收到关闭信号，开始优雅关闭...")

	// 注销服务
	if err := sm.registry.DeregisterService(sm.serviceConfig.ID); err != nil {
		log.Printf("❌ 服务注销失败: %v", err)
	}

	log.Println("✅ 服务已优雅关闭")
	os.Exit(0)
}

// 发现服务
func (sm *ServiceManager) DiscoverService(serviceName string) ([]*ServiceInstance, error) {
	return sm.registry.DiscoverService(serviceName)
}

// 等待关闭信号
func (sm *ServiceManager) waitForShutdown() {
	<-sm.stopChan
}
