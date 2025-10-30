package consul

import (
	"fmt"
	"free-chat/shared/config"

	"github.com/hashicorp/consul/api"
)

func Register(cfg config.ConsulConfig) error {
	consulCfg := api.DefaultConfig()
	consulCfg.Address = cfg.Address
	client, err := api.NewClient(consulCfg)
	if err != nil {
		return fmt.Errorf("创建Consul客户端失败: %v", err)
	}

	registration := &api.AgentServiceRegistration{
		ID:   ,
		Name: serviceName,
		Port: port,
		// Address: getOutboundIP(), // 或从 env 获取
		Check: &api.AgentServiceCheck{
			HTTP:     fmt.Sprintf("http://localhost:%d/health", port),
			Interval: "10s",
		},
	}
	client.Agent().ServiceRegister(registration)
}
