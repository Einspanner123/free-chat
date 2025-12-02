package persistence

import (
	"context"
	"fmt"
	"time"

	"free-chat/services/auth-service/internal/domain"

	"gorm.io/gorm"
)

// UserEntity 是数据库模型
type UserEntity struct {
	ID        uint      `gorm:"primaryKey;autoIncrement"`
	UserID    string    `gorm:"uniqueIndex;size:36;not null"`
	Username  string    `gorm:"uniqueIndex;size:50;not null"`
	Email     string    `gorm:"uniqueIndex;size:100;not null"`
	Password  string    `gorm:"size:255;not null"`
	CreatedAt time.Time `gorm:"autoCreateTime"`
	UpdatedAt time.Time `gorm:"autoUpdateTime"`
}

func (UserEntity) TableName() string {
	return "users"
}

func (u *UserEntity) ToDomain() *domain.User {
	return &domain.User{
		UserID:    u.UserID,
		Username:  u.Username,
		Email:     u.Email,
		Password:  u.Password,
		CreatedAt: u.CreatedAt,
		UpdatedAt: u.UpdatedAt,
	}
}

func FromDomain(u *domain.User) *UserEntity {
	return &UserEntity{
		UserID:   u.ID,
		Username: u.Username,
		Email:    u.Email,
		Password: u.Password,
	}
}

type UserRepository struct {
	db *gorm.DB
}

func NewUserRepository(db *gorm.DB) *UserRepository {
	return &UserRepository{db: db}
}

func (r *UserRepository) Save(ctx context.Context, user *domain.User) error {
	entity := FromDomain(user)
	if err := r.db.
		WithContext(ctx).
		Create(entity).Error; err != nil {
		return fmt.Errorf("failed to save user: %w", err)
	}
	return nil
}

func (r *UserRepository) FindByUsername(ctx context.Context, username string) (*domain.User, error) {
	var entity UserEntity
	if err := r.db.
		WithContext(ctx).
		Where("username = ?", username).
		First(&entity).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, domain.ErrUserNotFound
		}
		return nil, err
	}
	return entity.ToDomain(), nil
}

func (r *UserRepository) FindByEmail(ctx context.Context, email string) (*domain.User, error) {
	var entity UserEntity
	if err := r.db.
		WithContext(ctx).
		Where("email = ?", email).
		First(&entity).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, domain.ErrUserNotFound
		}
		return nil, err
	}
	return entity.ToDomain(), nil
}

func (r *UserRepository) FindByID(ctx context.Context, id string) (*domain.User, error) {
	var entity UserEntity
	if err := r.db.WithContext(ctx).
		Where("user_id = ?", id).
		First(&entity).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, domain.ErrUserNotFound
		}
		return nil, err
	}
	return entity.ToDomain(), nil
}
