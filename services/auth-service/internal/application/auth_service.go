package application

import (
	"context"
	"fmt"
	"time"

	"free-chat/services/auth-service/internal/domain"
)

type AuthService struct {
	userRepo        domain.UserRepository
	tokenService    domain.TokenService
	passwordService domain.PasswordService
}

func NewAuthService(
	userRepo domain.UserRepository,
	tokenService domain.TokenService,
) *AuthService {
	return &AuthService{
		userRepo:     userRepo,
		tokenService: tokenService,
	}
}

type LoginResult struct {
	AccessToken  string
	RefreshToken string
	ExpiresAt    time.Time
}

func (s *AuthService) Login(ctx context.Context, username, password string) (*LoginResult, error) {
	user, err := s.userRepo.FindByUsername(ctx, username)
	if err != nil {
		if err == domain.ErrUserNotFound {
			return nil, domain.ErrUserNotFound // Or generic invalid credentials
		}
		return nil, fmt.Errorf("failed to find user: %w", err)
	}

	if !s.passwordService.Compare(user.Password, password) {
		return nil, domain.ErrInvalidPassword
	}

	accessToken, accessExpireAt, err := s.tokenService.GenerateAccessToken(user.ID, user.Username)
	if err != nil {
		return nil, fmt.Errorf("failed to generate access token: %w", err)
	}

	refreshToken, _, err := s.tokenService.GenerateRefreshToken(user.ID, user.Username)
	if err != nil {
		return nil, fmt.Errorf("failed to generate refresh token: %w", err)
	}

	return &LoginResult{
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
		ExpiresAt:    accessExpireAt,
	}, nil
}

type RegisterResult struct {
	UserID string
}

func (s *AuthService) Register(ctx context.Context, username, email, password string) (*RegisterResult, error) {
	// Check if user already exists
	if _, err := s.userRepo.FindByUsername(ctx, username); err == nil {
		return nil, domain.ErrUserAlreadyExists
	}
	if _, err := s.userRepo.FindByEmail(ctx, email); err == nil {
		return nil, domain.ErrUserAlreadyExists
	}

	user, err := domain.NewUser(username, email, password)
	if err != nil {
		return nil, fmt.Errorf("failed to create user: %w", err)
	}

	if err := s.userRepo.Save(ctx, user); err != nil {
		return nil, fmt.Errorf("failed to save user: %w", err)
	}

	return &RegisterResult{
		UserID: user.ID,
	}, nil
}

func (s *AuthService) ValidateToken(ctx context.Context, token string) (*domain.TokenClaims, error) {
	return s.tokenService.ValidateToken(token)
}

func (s *AuthService) RefreshToken(ctx context.Context, refreshToken string) (string, time.Time, error) {
	return s.tokenService.RefreshToken(refreshToken)
}
