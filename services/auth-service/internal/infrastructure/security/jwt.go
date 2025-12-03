package security

import (
	"errors"
	"fmt"
	"free-chat/services/auth-service/internal/domain"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

const (
	TypeAccess  = "access"
	TypeRefresh = "refresh"
)

type JWTService struct {
	secretKey         string
	accessExpiration  time.Duration
	refreshExpiration time.Duration
}

type Claims struct {
	UserID   string `json:"user_id"`
	Username string `json:"username"`
	jwt.RegisteredClaims
}

func NewJWTService(secretKey string, expirationAccess, expirationRefresh int) *JWTService {
	return &JWTService{
		secretKey:         secretKey,
		accessExpiration:  time.Duration(expirationAccess) * time.Hour,
		refreshExpiration: time.Duration(expirationRefresh) * time.Hour,
	}
}

func (j *JWTService) GenerateAccessToken(userID string, userName string) (*domain.Token, error) {
	token, expiresAt, err := j.generate(userID, userName, TypeAccess)
	return &domain.Token{
		Token:     *token,
		ExpiresAt: *expiresAt,
	}, err
}

func (j *JWTService) GenerateRefreshToken(userID string, userName string) (*domain.Token, error) {
	token, expiresAt, err := j.generate(userID, userName, TypeRefresh)
	if err != nil {
		return nil, err
	}
	return &domain.Token{
		Token:     *token,
		ExpiresAt: *expiresAt,
	}, nil
}

func (j *JWTService) ValidateToken(tokenStr string) (bool, error) {
	token, err := jwt.ParseWithClaims(tokenStr, &Claims{},
		func(token *jwt.Token) (any, error) {
			if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
			}
			return []byte(j.secretKey), nil
		})
	if err != nil {
		return false, err
	}

	if _, ok := token.Claims.(*Claims); ok && token.Valid {
		return true, nil
	}

	return false, fmt.Errorf("invalid token")
}

func (j *JWTService) RefreshToken(tokenStr string) (*domain.Token, *domain.Token, error) {
	// Validate existing token
	claims := &Claims{}
	token, err := jwt.ParseWithClaims(tokenStr, claims,
		func(token *jwt.Token) (any, error) {
			return []byte(j.secretKey), nil
		})

	if err != nil || !token.Valid {
		return nil, nil, errors.New("invalid token")
	}

	if claims.Subject != TypeRefresh {
		return nil, nil, errors.New("invalid token type")
	}
	accessToken, err := j.GenerateAccessToken(claims.UserID, claims.Username)
	if err != nil {
		return nil, nil, err
	}
	refreshToken, err := j.GenerateRefreshToken(claims.UserID, claims.Username)
	if err != nil {
		return nil, nil, err
	}
	return accessToken, refreshToken, nil
}

func (j *JWTService) generate(userID, username, tokenType string) (*string, *time.Time, error) {
	claims := &Claims{
		UserID:   userID,
		Username: username,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(j.refreshExpiration)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
			Subject:   tokenType,
		},
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenStr, err := token.SignedString([]byte(j.secretKey))
	if err != nil {
		return nil, nil, err
	}
	return &tokenStr, &claims.ExpiresAt.Time, nil
}
