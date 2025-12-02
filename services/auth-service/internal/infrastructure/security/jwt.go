package security

import (
	"fmt"
	"time"

	"free-chat/services/auth-service/internal/domain"

	"github.com/golang-jwt/jwt/v5"
)

type JWTService struct {
	secretKey         string
	expirationAccess  time.Duration
	expirationRefresh time.Duration
}

type Claims struct {
	UserID   string `json:"user_id"`
	UserName string `json:"user_name"`
	jwt.RegisteredClaims
}

func NewJWTService(secretKey string, expirationAccess, expirationRefresh int) *JWTService {
	return &JWTService{
		secretKey:         secretKey,
		expirationAccess:  time.Duration(expirationAccess) * time.Hour,
		expirationRefresh: time.Duration(expirationRefresh) * time.Hour,
	}
}

func (j *JWTService) GenerateAccessToken(userID string, userName string) (string, time.Time, error) {
	claims := &Claims{
		UserID:   userID,
		UserName: userName,
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

func (j *JWTService) GenerateRefreshToken(userID string, userName string) (string, time.Time, error) {
	claims := &Claims{
		UserID:   userID,
		UserName: userName,
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

func (j *JWTService) ValidateToken(tokenStr string) (*domain.TokenClaims, error) {
	token, err := jwt.ParseWithClaims(tokenStr, &Claims{},
		func(token *jwt.Token) (any, error) {
			if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
			}
			return []byte(j.secretKey), nil
		})
	if err != nil {
		return nil, err
	}

	if claims, ok := token.Claims.(*Claims); ok && token.Valid {
		return &domain.TokenClaims{
			UserID:    claims.UserID,
			Username:  claims.UserName,
			ExpiresAt: claims.ExpiresAt.Time,
		}, nil
	}

	return nil, fmt.Errorf("invalid token")
}

func (j *JWTService) RefreshToken(tokenStr string) (string, time.Time, error) {
	// Validate existing token
	claims := &Claims{}
	token, err := jwt.ParseWithClaims(tokenStr, claims,
		func(token *jwt.Token) (any, error) {
			return []byte(j.secretKey), nil
		})

	if err != nil || !token.Valid {
		return "", time.Time{}, fmt.Errorf("invalid token")
	}

	if claims.Subject != "refresh" {
		return "", time.Time{}, fmt.Errorf("invalid token type")
	}

	// Generate new access token
	return j.GenerateAccessToken(claims.UserID, claims.UserName)
}
