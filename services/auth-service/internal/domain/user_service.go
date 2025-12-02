package domain

import (
	"regexp"
)

type UserDomainService struct{}

func NewUserDomainService() *UserDomainService {
	return &UserDomainService{}
}

func (s *UserDomainService) ValidateUserForCreate(u *User, repo UserRepository) error {
	if !regexp.
		MustCompile(`^[a-zA-Z0-9_]{1,20}$`).
		MatchString(u.Username) {
		return ErrInvalidUsername
	}
	if !regexp.
		MustCompile(`^[a-zA-Z0-9_@.]+$`).
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
