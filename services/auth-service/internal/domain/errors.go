package domain

import "errors"

// user
var (
	ErrUserAlreadyExists = errors.New("user already exists")
	ErrInvalidPassword   = errors.New("invalid password")
	ErrInvalidEmail      = errors.New("invalid email format")
	ErrInvalidUsername   = errors.New("username cannot be empty")
	ErrUserDisabled      = errors.New("user is disabled")
)

// repo
var (
	ErrUserSaveFailed = errors.New("failed to save user")
	ErrUserNotFound   = errors.New("user not found")
)

// token
var (
	ErrTokenGenerateFailed = errors.New("generate token failed")
)
