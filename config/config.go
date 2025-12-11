package config

import (
	"strings"
	"time"

	"github.com/spf13/viper"
)

// use viper instead
// $go get github.com/spf13/viper

type AppConfig struct {
	ServerName  string         `mapstructure:"server_name" yaml:"server_name"`
	Version     string         `mapstructure:"version" yaml:"version"`
	Environment string         `mapstructure:"environment" yaml:"environment"`
	Port        int            `mapstructure:"port" yaml:"port"`
	Mysql       MysqlConfig    `mapstructure:"mysql" yaml:"mysql"`
	Redis       RedisConfig    `mapstructure:"redis" yaml:"redis"`
	Consul      ConsulConfig   `mapstructure:"consul" yaml:"consul"`
	Postgres    PostgresConfig `mapstructure:"postgres" yaml:"postgres"`
	Chat        ChatConfig     `mapstructure:"chat" yaml:"chat"`
	Auth        AuthConfig     `mapstructure:"auth" yaml:"auth"`
	LLM         LLMConfig      `mapstructure:"llm" yaml:"llm"`
	RocketMQ    RocketMQConfig `mapstructure:"rocketmq" yaml:"rocketmq"`
}

type MysqlConfig struct {
	Host     string        `mapstructure:"host" yaml:"host"`
	Port     int           `mapstructure:"port" yaml:"port"`
	Username string        `mapstructure:"username" yaml:"username"`
	Password string        `mapstructure:"password" yaml:"password"`
	Database string        `mapstructure:"database" yaml:"database"`
	MaxIdle  int           `mapstructure:"max_idle" yaml:"max_idle"`
	MaxOpen  int           `mapstructure:"max_open" yaml:"max_open"`
	MaxLife  time.Duration `mapstructure:"max_life" yaml:"max_life"`
}

type RedisConfig struct {
	Address         string        `mapstructure:"address" yaml:"address"`
	Port            int           `mapstructure:"port" yaml:"port"`
	Password        string        `mapstructure:"password" yaml:"password"`
	Database        int           `mapstructure:"database" yaml:"database"`
	Prefix          string        `mapstructure:"prefix" yaml:"prefix"`
	DialTimeout     time.Duration `mapstructure:"dial_timeout" yaml:"dial_timeout"`
	ReadTimeout     time.Duration `mapstructure:"read_timeout" yaml:"read_timeout"`
	WriteTimeout    time.Duration `mapstructure:"write_timeout" yaml:"write_timeout"`
	MaxRetries      int           `mapstructure:"max_retries" yaml:"max_retries"`
	PoolSize        int           `mapstructure:"pool_size" yaml:"pool_size"`
	MinIdleConns    int           `mapstructure:"min_idle_conns" yaml:"min_idle_conns"`
	CacheTTL        time.Duration `mapstructure:"cache_ttl" yaml:"cache_ttl"`
	CacheJitterSec  int           `mapstructure:"cache_jitter_sec" yaml:"cache_jitter_sec"`
	LockTTL         time.Duration `mapstructure:"lock_ttl" yaml:"lock_ttl"`
	LockMaxAttempts int           `mapstructure:"lock_max_attempts" yaml:"lock_max_attempts"`
	LockBackoff     time.Duration `mapstructure:"lock_backoff" yaml:"lock_backoff"`
	RateLimitQPS    int           `mapstructure:"rate_limit_qps" yaml:"rate_limit_qps"`
}

type PostgresConfig struct {
	Address  string        `mapstructure:"address" yaml:"address"`
	Port     int           `mapstructure:"port" yaml:"port"`
	User     string        `mapstructure:"user" yaml:"user"`
	Password string        `mapstructure:"password" yaml:"password"`
	DBName   string        `mapstructure:"db_name" yaml:"db_name"`
	MaxIdle  int           `mapstructure:"max_idle" yaml:"max_idle"`
	MaxOpen  int           `mapstructure:"max_open" yaml:"max_open"`
	MaxLife  time.Duration `mapstructure:"max_life" yaml:"max_life"`
}

type ConsulConfig struct {
	Address    string `mapstructure:"address" yaml:"address"`
	Scheme     string `mapstructure:"scheme" yaml:"scheme"`
	Datacenter string `mapstructure:"datacenter" yaml:"datacenter"`
}

type ChatConfig struct {
	ServerName string `mapstructure:"server_name" yaml:"server_name"`
	GRPCPort   int    `mapstructure:"grpc_port" yaml:"grpc_port"`
}

type AuthConfig struct {
	ServerName       string `mapstructure:"server_name" yaml:"server_name"`
	GRPCPort         int    `mapstructure:"grpc_port" yaml:"grpc_port"`
	JwtSecret        string `mapstructure:"jwt_secret" yaml:"jwt_secret"`
	Expire_Access_H  int    `mapstructure:"expire_access_h" yaml:"expire_access_h"`
	Expire_Refresh_H int    `mapstructure:"expire_refresh_h" yaml:"expire_refresh_h"`
}

type LLMConfig struct {
	Name        string   `mapstructure:"name" yaml:"name"`
	Port        int      `mapstructure:"port" yaml:"port"`
	Models      []string `mapstructure:"models" yaml:"models"`
	Temperature float64  `mapstructure:"temperature" yaml:"temperature"`
	TopP        float64  `mapstructure:"top_p" yaml:"top_p"`
	MaxTokens   int      `mapstructure:"max_tokens" yaml:"max_tokens"`
}

type RocketMQConfig struct {
	NameServers   []string `mapstructure:"name_servers" yaml:"name_servers"`
	MaxRetries    int      `mapstructure:"max_retries" yaml:"max_retries"`
	GroupName     string   `mapstructure:"group_name" yaml:"group_name"`
	ConsumerGroup string   `mapstructure:"consumer_group" yaml:"consumer_group"`
	MessageModel  string   `mapstructure:"message_model" yaml:"message_model"`
	Topics        struct {
		LLMRequest  string `mapstructure:"llm_request" yaml:"llm_request"`
		LLMResponse string `mapstructure:"llm_response" yaml:"llm_response"`

		UserEvent string `mapstructure:"user_event" yaml:"user_event"`
		SystemLog string `mapstructure:"system_log" yaml:"system_log"`
	} `mapstructure:"topics" yaml:"topics"`
}

func LoadConfig() (*AppConfig, error) {
	var config AppConfig

	viper.SetConfigFile("config/config.yml")
	viper.SetConfigType("yaml")

	viper.AutomaticEnv()
	viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))

	if err := viper.ReadInConfig(); err != nil {
		return &config, err
	}
	if err := viper.Unmarshal(&config); err != nil {
		return &config, err
	}
	return &config, nil
}
