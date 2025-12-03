package dto

// RegisterReq 注册请求DTO
type RegisterReq struct {
	Username string `json:"username" binding:"required"` // Gin参数校验
	Password string `json:"password" binding:"required,min=6"`
	Email    string `json:"email" binding:"required,email"`
}

// RegisterResp 注册响应DTO
type RegisterResp struct {
	UserID   string `json:"user_id"`
	Username string `json:"username"`
	Email    string `json:"email"`
}

// LoginReq 登录请求DTO
type LoginReq struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
}

// LoginResp 登录响应DTO（含JWT Token）
type LoginResp struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresAt    int64  `json:"expires_at"` // Token过期时间（秒）
	UserID       string `json:"user_id"`
}
