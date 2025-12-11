package domain

import (
	"regexp"
)

type UserService struct{}

func NewUserService() *UserService {
	return &UserService{}
}

func (s *UserService) ValidateForCreate(u *User, repo UserRepository) error {
	if !regexp.
		MustCompile(`^[a-zA-Z0-9_]{1,20}$`).
		MatchString(u.Username) {
		return ErrInvalidUsername
	}
	if !regexp.
		MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`).
		MatchString(u.Email) {
		return ErrInvalidEmail
	}
	existUser, _ := repo.FindByUsername(u.Username)
	if existUser != nil {
		return ErrUserAlreadyExists
	}
	existEmailUser, _ := repo.FindByEmail(u.Email)
	if existEmailUser != nil {
		return ErrUserAlreadyExists
	}

	return nil
}

type PasswordEncoder interface {
	Hash(raw string) (Password, error)
	Compare(hashedPassword, password string) bool
}

type TokenService interface {
	GenerateAccessToken(userID, username string) (*Token, error)
	GenerateRefreshToken(userID, username string) (*Token, error)
	ValidateToken(token string) (bool, error)
	RefreshToken(refreshToken string) (*Token, *Token, error)
}
