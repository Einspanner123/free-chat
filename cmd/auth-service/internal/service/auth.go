package service

import (
	"errors"
	"fmt"
	"free-chat/cmd/auth-service/internal/store"
	"free-chat/shared/config"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/crypto/bcrypt"
)

type AuthService struct {
	cfg   config.AuthConfig
	store *store.PostgresStore
}

// NewAuthService 初始化认证服务
func NewAuthService(cfg config.AuthConfig, store *store.PostgresStore) *AuthService {
	return &AuthService{
		cfg:   cfg,
		store: store,
	}
}

// 登录用户，校验token
func (s *AuthService) Login(username, password string) (token string, userID string, err error) {
	user, err := s.store.GetUserByUsername(username)
	if err != nil {
		return "", "", fmt.Errorf("登录失败: %v", err)
	}

	if err := bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(password)); err != nil {
		return "", "", errors.New("密码错误")
	}

	token, err = s.generateJWT(user.ID)
	if err != nil {
		return "", "", fmt.Errorf("生成令牌失败: %v", err)
	}

	return token, user.ID, nil
}

// VerifyToken 验证JWT令牌有效性，返回用户ID
func (s *AuthService) VerifyToken(tokenStr string) (userID string, err error) {
	// 解析令牌
	token, err := jwt.Parse(tokenStr, func(t *jwt.Token) (interface{}, error) {
		// 验证签名方法
		if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, jwt.ErrSignatureInvalid
		}
		return []byte(s.cfg.JwtSecret), nil
	})
	if err != nil {
		return "", fmt.Errorf("令牌无效: %v", err)
	}

	// 验证令牌有效性
	if !token.Valid {
		return "", errors.New("令牌无效或已过期")
	}

	// 提取用户ID
	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok {
		return "", errors.New("令牌格式错误")
	}
	userID, ok = claims["user_id"].(string)
	if !ok {
		return "", errors.New("令牌缺少user_id")
	}

	return userID, nil
}

// generateJWT 生成JWT令牌
func (s *AuthService) generateJWT(userID string) (string, error) {
	// 设置过期时间
	expireAt := time.Now().Add(time.Duration(s.cfg.Expire_H) * time.Hour).Unix()

	// 创建claims
	claims := jwt.MapClaims{
		"user_id": userID,
		"exp":     expireAt,          // 过期时间（Unix时间戳）
		"iat":     time.Now().Unix(), // 签发时间
	}

	// 生成令牌
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(s.cfg.JwtSecret))
}
