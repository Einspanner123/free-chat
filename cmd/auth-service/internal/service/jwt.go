package service

import (
	"fmt"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

type JWTService struct {
	secretKey         string
	expirationAccess  time.Duration
	expirationRefresh time.Duration
}

type Claims struct {
	UserID   int    `json:"user_id"`
	Username string `json:"username"`
	jwt.RegisteredClaims
}

func NewJWTService(secretKey string, expirationAccess, expirationRefresh int) *JWTService {
	return &JWTService{
		secretKey:         secretKey,
		expirationAccess:  time.Duration(expirationAccess) * time.Hour,
		expirationRefresh: time.Duration(expirationRefresh) * time.Hour,
	}
}

func (j *JWTService) GenerateAccessToken(userID int, username string) (string, time.Time, error) {
	claims := &Claims{
		UserID:   userID,
		Username: username,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(j.expirationAccess)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
			Subject:   "access",
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenStr, err := token.SignedString([]byte(j.secretKey))
	return tokenStr, claims.ExpiresAt.Time, err
}

func (j *JWTService) GenerateRefreshToken(userID int, username string) (string, time.Time, error) {
	claims := &Claims{
		UserID:   userID,
		Username: username,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(j.expirationRefresh)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
			Subject:   "refresh",
		},
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenStr, err := token.SignedString([]byte(j.secretKey))
	return tokenStr, claims.ExpiresAt.Time, err
}

func (j *JWTService) ValidateToken(tokenStr string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenStr, &Claims{},
		func(token *jwt.Token) (interface{}, error) {
			if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
			}
			return []byte(j.secretKey), nil
		})
	if err != nil {
		return nil, err
	}

	if claims, ok := token.Claims.(*Claims); ok && token.Valid {
		return claims, nil
	}

	return nil, fmt.Errorf("invalid token")
}

func (j *JWTService) RefreshToken(tokenStr string) (string, time.Time, error) {
	claims, err := j.ValidateToken(tokenStr)
	if err != nil {
		return "", time.Time{}, err
	}

	if claims.Subject != "refresh" {
		return "", time.Time{}, fmt.Errorf("invalid token type")
	}

	// 生成新令牌
	return j.GenerateAccessToken(claims.UserID, claims.Username)
}
