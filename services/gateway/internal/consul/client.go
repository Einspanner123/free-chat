package consul

import (
	"fmt"
	"free-chat/shared/config"

	"github.com/hashicorp/consul/api"
)

type ConsulRegistry struct {
	client *api.Client
	config *config.ConsulConfig
}

type Client struct {
	client *api.Client
}

func NewClient(cfg *config.ConsulConfig) (*Client, error) {
	config := api.DefaultConfig()
	config.Address = cfg.Address
	config.Datacenter = cfg.Datacenter
	config.Scheme = cfg.Scheme
	client, err := api.NewClient(config)
	if err != nil {
		return nil, fmt.Errorf("创建Consul客户端失败: %v", err)
	}
	return &Client{client: client}, nil
}

func (c *Client) DiscoverService(serviceName string) (string, error) {
	services, _, err := c.client.Health().Service(serviceName, "", true, nil)
	if err != nil {
		return "", fmt.Errorf("查询服务%s失败: %v", serviceName, err)
	}
	if len(services) == 0 {
		return "", fmt.Errorf("服务%s无健康实例", serviceName)
	}

	instance := services[0].Service
	return fmt.Sprintf("%s:%d", instance.Address, instance.Port), nil
}
