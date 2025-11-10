package middleware

import (
	"context"
	"free-chat/shared/proto/auth"
	"free-chat/shared/registry"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func JwtAuth(consul *registry.ConsulRegistry, authServiceName string) gin.HandlerFunc {
	return func(c *gin.Context) {
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "缺少Authorization头",
			})
			c.Abort()
			return
		}
		parts := strings.Split(authHeader, " ")
		if len(parts) != 2 || parts[0] != "Bearer" {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "Authorization格式错误",
			})
			c.Abort()
			return
		}
		token := parts[1]
		// token, err := jwt.Parse(parts[1], func(t *jwt.Token) (interface{}, error) {
		// 	if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
		// 		return nil, jwt.ErrSignatureInvalid
		// 	}
		// 	return []byte(jwtSecret), nil
		// })
		// if err != nil || !token.Valid {
		// 	c.JSON(http.StatusUnauthorized, gin.H{
		// 		"error": "无效的令牌",
		// 	})
		// 	c.Abort()
		// 	return
		// }
		instances, err := consul.DiscoverService(authServiceName)
		if err != nil || len(instances) == 0 {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "认证服务不可用"})
			c.Abort()
			return
		}
		// 注意：GetURL 返回 "http://host:port"；若你的 gRPC 版本不接受 scheme，改为 host:port
		conn, err := grpc.NewClient(instances[0].GetURL(), grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "连接认证服务失败"})
			c.Abort()
			return
		}
		defer conn.Close()

		client := auth.NewAuthServiceClient(conn)
		ctx, cancel := context.WithTimeout(c.Request.Context(), 3*time.Second)
		defer cancel()

		resp, err := client.VerifyToken(ctx, &auth.VerifyTokenRequest{JwtToken: token})
		if err != nil || !resp.Valid {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "令牌无效"})
			c.Abort()
			return
		}
		// claims, ok := token.Claims.(jwt.MapClaims)
		// if !ok {
		// 	c.JSON(http.StatusUnauthorized, gin.H{"error": "令牌格式错误"})
		// 	c.Abort()
		// 	return
		// }
		// userID, ok := claims["user_id"].(string)
		// if !ok {
		// 	c.JSON(http.StatusUnauthorized, gin.H{"error": "令牌缺少user_id"})
		// 	c.Abort()
		// 	return
		// }
		c.Set("user_id", resp.UserId)
		c.Next()
	}
}
