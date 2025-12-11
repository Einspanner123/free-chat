package db

import (
	"free-chat/services/auth-service/internal/domain"
	"time"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

func InitGorm(dsn string) (*gorm.DB, error) {
	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
		// 生产级配置
	})
	if err != nil {
		return nil, err
	}
	err = db.AutoMigrate(&UserModel{})
	if err != nil {
		return nil, err
	}
	return db, nil
}

// UserModel 数据库映射模型（与领域实体User分离，避免领域层依赖GORM）
type UserModel struct {
	ID        string    `gorm:"primaryKey;type:varchar(36)"`
	Username  string    `gorm:"uniqueIndex;type:varchar(20);not null"`
	Password  string    `gorm:"type:varchar(100);not null"` // 存储密码哈希
	Email     string    `gorm:"uniqueIndex;type:varchar(100);not null"`
	Status    int8      `gorm:"type:smallint;default:1"` // 映射UserStatus
	CreatedAt time.Time `gorm:"autoCreateTime"`
	UpdatedAt time.Time `gorm:"autoUpdateTime"`
}

// 转换：UserModel → 领域实体User
func (m *UserModel) ToDomainEntity() *domain.User {
	return &domain.User{
		ID:       m.ID,
		Username: m.Username,
		Password: *domain.NewPassword(m.Password), // 直接赋值哈希（避免二次加密）
		Email:    m.Email,
		Status:   domain.UserStatus(m.Status),
	}
}

// 转换：领域实体User → UserModel
func ToUserModel(d *domain.User) *UserModel {
	return &UserModel{
		ID:       d.ID,
		Username: d.Username,
		Password: d.Password.Hash(), // 需给Password加Hash()方法暴露哈希值
		Email:    d.Email,
		Status:   int8(d.Status),
	}
}
