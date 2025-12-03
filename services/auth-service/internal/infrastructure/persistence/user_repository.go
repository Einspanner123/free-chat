package persistence

import (
	"time"

	"free-chat/services/auth-service/internal/domain"
	"free-chat/services/auth-service/internal/infrastructure/db"

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

type UserRepository struct {
	db *gorm.DB
}

func NewUserRepository(db *gorm.DB) *UserRepository {
	return &UserRepository{db: db}
}

func (r *UserRepository) Save(user *domain.User) error {
	model := db.ToUserModel(user)
	return r.db.Create(model).Error
}

func (r *UserRepository) FindByUsername(username string) (*domain.User, error) {
	var model db.UserModel
	if err := r.db.
		Where("username = ?", username).
		First(&model).Error; err != nil {
		return nil, err
	}
	return model.ToDomainEntity(), nil
}

func (r *UserRepository) FindByEmail(email string) (*domain.User, error) {
	var model db.UserModel
	if err := r.db.
		Where("email = ?", email).
		First(&model).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, domain.ErrUserNotFound
		}
		return nil, err
	}
	return model.ToDomainEntity(), nil
}

func (r *UserRepository) FindByID(id string) (*domain.User, error) {
	var model db.UserModel
	if err := r.db.
		Where("user_id = ?", id).
		First(&model).Error; err != nil {
		return nil, err
	}
	return model.ToDomainEntity(), nil
}
