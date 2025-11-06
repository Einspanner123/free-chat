package store

import (
	"database/sql"
	"fmt"
	"free-chat/shared/config"
)

type User struct {
	ID       string
	Username string
	Password string
}

type PostgresStore struct {
	db *sql.DB
}

func NewPostgresStore(cfg config.PostgresConfig) (*PostgresStore, error) {
	connStr := fmt.Sprintf(
		"host=%s port=%s user=%s password=%s dbname=%s sslmode=disable",
		cfg.Address, cfg.Port, cfg.User, cfg.Password, cfg.DBName,
	)
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, fmt.Errorf("打开数据库失败: %v", err)
	}
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("连接数据库失败: %v", err)
	}
	return &PostgresStore{db: db}, nil
}

func (s *PostgresStore) GetUserByUsername(username string) (*User, error) {
	query := "SELECT id, username, password FROM users WHERE username = $1 LIMIT 1"
	row := s.db.QueryRow(query, username)

	var user User
	err := row.Scan(&user.ID, &user.Username, &user.Password)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("用户不存在")
	}
	if err != nil {
		return nil, fmt.Errorf("查询用户失败: %v", err)
	}
	return &user, nil
}

func (s *PostgresStore) Close() error {
	return s.db.Close()
}
