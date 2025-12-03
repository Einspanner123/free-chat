package security

import (
	"free-chat/services/auth-service/internal/domain"

	"golang.org/x/crypto/bcrypt"
)

type BcryptService struct{}

func NewBcryptService() *BcryptService {
	return &BcryptService{}
}

func (s *BcryptService) Hash(plain string) (domain.Password, error) {
	bytes, err := bcrypt.GenerateFromPassword([]byte(plain), bcrypt.DefaultCost)
	password := domain.NewPassword(string(bytes))
	return *password, err
}

func (s *BcryptService) Compare(hash, plain string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(plain))
	return err == nil
}
