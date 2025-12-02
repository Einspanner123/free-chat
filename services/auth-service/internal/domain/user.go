package domain

import (
	"context"
	"errors"
	"time"
)

var (
	ErrUserNotFound      = errors.New("user not found")
	ErrUserAlreadyExists = errors.New("user already exists")
	ErrInvalidPassword   = errors.New("invalid password")
)

// User 是核心领域对象
// 纯 Go struct，不包含数据库标签
type User struct {
	ID        string
	UserID    string
	Username  string
	Email     string
	Password  string // 加密后的密码
	CreatedAt time.Time
	UpdatedAt time.Time
}

type UserRepository interface {
	Save(ctx context.Context, user *User) error
	FindByUsername(ctx context.Context, username string) (*User, error)
	FindByEmail(ctx context.Context, email string) (*User, error)
	FindByID(ctx context.Context, id string) (*User, error)
}

type PasswordService interface {
	Hash(password string) (string, error)
	Compare(hashedPassword, password string) bool
}

type TokenService interface {
	GenerateAccessToken(userID, username string) (string, time.Time, error)
	GenerateRefreshToken(userID, username string) (string, time.Time, error)
	ValidateToken(token string) (*TokenClaims, error)
	RefreshToken(refreshToken string) (string, time.Time, error)
}

type TokenClaims struct {
	UserID    string
	Username  string
	ExpiresAt time.Time
}
