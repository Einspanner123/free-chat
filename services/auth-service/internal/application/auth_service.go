package application

import (
	"free-chat/services/auth-service/internal/application/dto"
	"free-chat/services/auth-service/internal/domain"

	"github.com/google/uuid"
)

type AuthService struct {
	userService     domain.UserService
	userRepo        domain.UserRepository
	tokenService    domain.TokenService
	passwordService domain.PasswordEncoder
}

func NewAuthService(
	userService domain.UserService,
	userRepo domain.UserRepository,
	tokenService domain.TokenService,
	passwordService domain.PasswordEncoder,
) *AuthService {
	return &AuthService{
		userService:     userService,
		userRepo:        userRepo,
		tokenService:    tokenService,
		passwordService: passwordService,
	}
}

func (s *AuthService) Login(req *dto.LoginReq) (*dto.LoginResp, error) {
	u, err := s.userRepo.FindByUsername(req.Username)
	if err != nil {
		return nil, domain.ErrUserNotFound
	}
	if !u.Password.Verify(req.Password, s.passwordService) {
		return nil, domain.ErrInvalidPassword
	}
	if u.Status == domain.UserStatusDisabled {
		return nil, domain.ErrUserDisabled
	}
	accessToken, err := s.tokenService.GenerateAccessToken(u.ID, u.Username)
	if err != nil {
		return nil, domain.ErrTokenGenerateFailed
	}
	refreshToken, err := s.tokenService.GenerateRefreshToken(u.ID, u.Username)
	if err != nil {
		return nil, domain.ErrTokenGenerateFailed
	}
	return &dto.LoginResp{
		AccessToken:  accessToken.Token,
		RefreshToken: refreshToken.Token,
		ExpiresAt:    accessToken.ExpiresAt.Unix(),
		UserID:       u.ID,
	}, nil
}

func (s *AuthService) Register(req *dto.RegisterReq) (*dto.RegisterResp, error) {
	hashedPassword, err := s.passwordService.Hash(req.Password)
	if err != nil {
		return nil, err
	}
	newUser := domain.NewUser(
		uuid.NewString(), req.Username, req.Email, hashedPassword,
	)
	if err := s.userService.ValidateForCreate(newUser, s.userRepo); err != nil {
		return nil, err
	}

	if err := s.userRepo.Save(newUser); err != nil {
		return nil, err
	}
	return &dto.RegisterResp{
		UserID:   newUser.ID,
		Username: newUser.Username,
		Email:    newUser.Email,
	}, nil
}

func (s *AuthService) ValidateToken(token string) (bool, error) {
	return s.tokenService.ValidateToken(token)
}

func (s *AuthService) RefreshToken(refreshToken string) (*dto.LoginResp, error) {
	accessToken, newRefreshToken, err := s.tokenService.RefreshToken(refreshToken)
	if err != nil {
		return nil, err
	}
	return &dto.LoginResp{
		AccessToken:  accessToken.Token,
		RefreshToken: newRefreshToken.Token,
		ExpiresAt:    accessToken.ExpiresAt.Unix(),
	}, nil
}
