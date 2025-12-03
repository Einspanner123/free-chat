package domain

import "time"

type UserStatus int8

const (
	UserStatusActive UserStatus = iota
	UserStatusInactive
	UserStatusDisabled
)

type User struct {
	ID       string
	Username string
	Email    string
	Password Password
	Status   UserStatus
}

func NewUser(id, username, email string, password Password) *User {
	return &User{
		ID:       id,
		Username: username,
		Email:    email,
		Password: password,
		Status:   UserStatusActive,
	}
}

type Password struct {
	hash string
}

func NewPassword(hash string) *Password {
	return &Password{hash: hash}
}

func (p Password) Verify(raw string, encoder PasswordEncoder) bool {
	return encoder.Compare(p.hash, raw)
}

func (p Password) Hash() string {
	return p.hash
}

type Token struct {
	Token     string
	ExpiresAt time.Time
}
