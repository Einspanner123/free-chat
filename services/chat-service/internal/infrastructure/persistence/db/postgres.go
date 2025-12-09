package db

import (
	"free-chat/services/chat-service/internal/infrastructure/persistence/model"

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
	err = db.AutoMigrate(&model.MessageModel{}, &model.SessionModel{})
	if err != nil {
		return nil, err
	}
	return db, nil
}
