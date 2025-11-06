package registry

import (
	"fmt"
	"log"
	"net"
	"time"

	"github.com/hashicorp/consul/api"
)

type ConsulRegistry struct {
	client *api.Client
	config *ConsulConfig
}

type ConsulConfig struct {
	Address    string
	Scheme     string
	Datacenter string
}

type ServiceConfig struct {
	ID          string
	Name        string
	Tags        []string
	Address     string
	Port        int
	HealthCheck *HealthCheck
}

type HealthCheck struct {
	HTTP                           string
	Interval                       time.Duration
	Timeout                        time.Duration
	DeregisterCriticalServiceAfter time.Duration
}

// 创建Consul
func NewConsulRegistry(config *ConsulConfig) (*ConsulRegistry, error) {
	consulConfig := api.DefaultConfig()
	consulConfig.Address = config.Address
	consulConfig.Scheme = config.Scheme
	consulConfig.Datacenter = config.Datacenter

	client, err := api.NewClient(consulConfig)
	if err != nil {
		return nil, fmt.Errorf("创建Consul客户端失败: %v", err)
	}
	_, err = client.Status().Leader()
	if err != nil {
		return nil, fmt.Errorf("连接Consul失败: %v", err)
	}
	log.Printf("✅ Consul连接成功: %s", config.Address)
	return &ConsulRegistry{
		client: client,
		config: config,
	}, nil
}

// 注册服务
func (r *ConsulRegistry) RegisterService(config *ServiceConfig) error {
	// 构建服务注册信息
	registration := &api.AgentServiceRegistration{
		ID:      config.ID,
		Name:    config.Name,
		Tags:    config.Tags,
		Address: config.Address,
		Port:    config.Port,
	}

	// 添加健康检查
	if config.HealthCheck != nil {
		registration.Check = &api.AgentServiceCheck{
			HTTP:                           config.HealthCheck.HTTP,
			Interval:                       config.HealthCheck.Interval.String(),
			Timeout:                        config.HealthCheck.Timeout.String(),
			DeregisterCriticalServiceAfter: config.HealthCheck.DeregisterCriticalServiceAfter.String(),
		}
	}

	// 注册服务
	err := r.client.Agent().ServiceRegister(registration)
	if err != nil {
		return fmt.Errorf("服务注册失败: %v", err)
	}

	log.Printf("✅ 服务注册成功: %s (ID: %s)", config.Name, config.ID)
	return nil
}

// 注销服务
func (r *ConsulRegistry) DeregisterService(serviceID string) error {
	err := r.client.Agent().ServiceDeregister(serviceID)
	if err != nil {
		return fmt.Errorf("服务注销失败: %v", err)
	}

	log.Printf("✅ 服务注销成功: %s", serviceID)
	return nil
}

// 发现服务
func (r *ConsulRegistry) DiscoverService(serviceName string) ([]*ServiceInstance, error) {
	// 查询健康的服务实例
	services, _, err := r.client.Health().Service(serviceName, "", true, nil)
	if err != nil {
		return nil, fmt.Errorf("服务发现失败: %v", err)
	}

	var instances []*ServiceInstance
	for _, service := range services {
		instance := &ServiceInstance{
			ID:      service.Service.ID,
			Name:    service.Service.Service,
			Address: service.Service.Address,
			Port:    service.Service.Port,
			Tags:    service.Service.Tags,
		}
		instances = append(instances, instance)
	}

	return instances, nil
}

// 服务实例信息
type ServiceInstance struct {
	ID      string
	Name    string
	Address string
	Port    int
	Tags    []string
}

// // 获取服务URL
// func (s *ServiceInstance) GetURL() string {
// 	return fmt.Sprintf("http://%s:%d", s.Address, s.Port)
// }

// 获取本机IP地址
func GetLocalIP() (string, error) {
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err != nil {
		return "", err
	}
	defer conn.Close()

	localAddr := conn.LocalAddr().(*net.UDPAddr)
	return localAddr.IP.String(), nil
}

// 生成服务ID
func GenerateServiceID(serviceName string, port int) string {
	ip, _ := GetLocalIP()
	return fmt.Sprintf("%s-%s-%d", serviceName, ip, port)
}

// consul/
// ├── client.go           # 基础客户端
// │   ├── NewClient()     # 创建连接
// │   ├── GetClient()     # 获取客户端
// │   └── Close()         # 关闭连接
// │
// ├── registry.go         # 服务注册
// │   ├── Register()      # 注册服务
// │   ├── Deregister()    # 注销服务
// │   └── UpdateHealth()  # 更新健康状态
// │
// └── discovery.go        # 服务发现
//     ├── Discover()      # 发现服务
//     ├── Watch()         # 监听变化
//     └── LoadBalance()   # 负载均衡
