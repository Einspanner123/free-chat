package domain

import (
	"errors"
	"time"
)

var (
	ErrUserNotFound      = errors.New("user not found")
	ErrUserAlreadyExists = errors.New("user already exists")
	ErrInvalidPassword   = errors.New("invalid password")
	ErrInvalidEmail      = errors.New("invalid email format")
	ErrInvalidUsername   = errors.New("username cannot be empty")
)

type UserStatus int8

const (
	UserStatusActive    UserStatus = 1
	UserStatusInactive  UserStatus = 2
	UserStatusSuspended UserStatus = 3
)

// User 是聚合根 (Aggregate Root)
// 包含业务逻辑和不变量检查
type User struct {
	ID       string
	Username string
	Email    string
	Password string // 加密后的密码
	Status   UserStatus
}

type Password struct {
	hash string
}

func (p Password) Verify(plainPassword string, h PasswordHandler) bool {
	return h.Compare(p.hash, plainPassword)
}

type PasswordHandler interface {
	Hash(password string) (string, error)
	Compare(hashedPassword, password string) bool
}

type TokenClaims struct {
	Token     string
	ExpiresAt time.Time
}

type TokenService interface {
	GenerateAccessToken(userID, username string) (*TokenClaims, error)
	GenerateRefreshToken(userID, username string) (*TokenClaims, error)
	ValidateToken(token string) (bool, error)
	RefreshToken(refreshToken string) (*TokenClaims, *TokenClaims, error)
}
