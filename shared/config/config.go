package config

import (
	"os"
	"strconv"
	"time"
)

// use viper instead
// $go get github.com/spf13/viper

type AppConfig struct {
	ServerName  string
	Version     string
	Environment string
	Port        int

	// server
	Mysql    MysqlConfig
	Redis    RedisConfig
	Consul   ConsulConfig
	Postgres PostgresConfig
	Chat     ChatConfig
	Auth     AuthConfig
	LLM      LLMConfig
}

type MysqlConfig struct {
	Host     string
	Port     int
	Username string
	Password string
	Database string
	MaxIdle  int
	MaxOpen  int
	MaxLife  time.Duration
}

type RedisConfig struct {
	Address      string
	Port         int
	Password     string
	Database     int
	RateLimitQPS int
}

type PostgresConfig struct {
	Address  string
	Port     int
	User     string
	Password string
	DBName   string
}

type ConsulConfig struct {
	Address    string
	Scheme     string
	Datacenter string
}

type ChatConfig struct {
	ServerName string
}

type AuthConfig struct {
	ServerName string
	GRPCPort   int
	// Jwt
	JwtSecret        string
	Expire_Access_H  int
	Expire_Refresh_H int
}

type LLMConfig struct {
	Models      []string
	Temperature float64
	TopP        float64
	MaxTokens   int
}

func LoadConfig(serviceName string) *AppConfig {
	return &AppConfig{
		ServerName:  serviceName,
		Version:     getEnv("APP_VERSION", "1.0.0"),
		Environment: getEnv("APP_ENV", "development"),
		Port:        getEnvInt("APP_PORT", 8080),

		Mysql: MysqlConfig{
			Host:     getEnv("MYSQL_HOST", "localhost"),
			Port:     getEnvInt("MYSQL_PORT", 3306),
			Username: getEnv("MYSQL_USERNAME", "root"),
			Password: getEnv("MYSQL_PASSWORD", "123456"),
			Database: getEnv("MYSQL_DATABASE", "free-chat"),
			MaxIdle:  getEnvInt("MYSQL_MAX_IDLE", 10),
			MaxOpen:  getEnvInt("MYSQL_MAX_OPEN", 100),
			MaxLife:  time.Duration(getEnvInt("MYSQL_MAX_LIFE", 3600)) * time.Second,
		},

		Redis: RedisConfig{
			Address:      getEnv("REDIS_ADDR", "localhost"),
			Port:         getEnvInt("REDIS_PORT", 6379),
			Password:     getEnv("REDIS_PASSWORD", ""),
			Database:     getEnvInt("REDIS_DATABASE", 0),
			RateLimitQPS: getEnvInt("RATE_LIMIT_QPS", 10),
		},

		Postgres: PostgresConfig{
			Address:  getEnv("PG_ADDR", "localhost"),
			Port:     getEnvInt("PG_PORT", 5432),
			User:     getEnv("PG_USER", "free-chat"),
			Password: getEnv("PG_PASSWD", "free-chat-passwd"),
			DBName:   getEnv("PG_DBNAME", "free-chat"),
		},

		Consul: ConsulConfig{
			Address:    getEnv("CONSUL_ADDRESS", "localhost:8500"),
			Scheme:     getEnv("CONSUL_SCHEME", "http"),
			Datacenter: getEnv("CONSUL_DATACENTER", "dc1"),
		},

		Chat: ChatConfig{
			ServerName: getEnv("CHAT_SERVICE_NAME", "chat-service"),
		},

		Auth: AuthConfig{
			ServerName:       getEnv("AUTH_SERVICE_NAME", "auth-service"),
			GRPCPort:         getEnvInt("GRPC_PORT", 8083),
			JwtSecret:        "llm_chat_secret",
			Expire_Access_H:  1,
			Expire_Refresh_H: 24 * 3,
		},
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvFloat(key string, defaultValue float64) float64 {
	if value := os.Getenv(key); value != "" {
		if floatValue, err := strconv.ParseFloat(value, 64); err == nil {
			return floatValue
		}
	}
	return defaultValue
}
