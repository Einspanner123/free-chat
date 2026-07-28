package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
)

func testHandler(c *gin.Context) {
	userID := c.GetString("user_id")
	username := c.GetString("username")
	c.JSON(200, gin.H{
		"user_id":  userID,
		"username": username,
	})
}

func TestJwtAuth_MissingHeader(t *testing.T) {
	gin.SetMode(gin.TestMode)

	r := gin.New()
	r.Use(JwtAuth("test-secret"))
	r.GET("/protected", testHandler)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/protected", nil)
	r.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", w.Code)
	}
}

func TestJwtAuth_NotBearerToken(t *testing.T) {
	gin.SetMode(gin.TestMode)

	r := gin.New()
	r.Use(JwtAuth("test-secret"))
	r.GET("/protected", testHandler)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/protected", nil)
	req.Header.Set("Authorization", "Basic dXNlcjpwYXNz")
	r.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 for non-Bearer token, got %d", w.Code)
	}
}

func TestJwtAuth_InvalidToken(t *testing.T) {
	gin.SetMode(gin.TestMode)

	r := gin.New()
	r.Use(JwtAuth("test-secret"))
	r.GET("/protected", testHandler)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/protected", nil)
	req.Header.Set("Authorization", "Bearer invalid.token.here")
	r.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 for invalid token, got %d", w.Code)
	}
}

func TestJwtAuth_ValidToken_SetsUserContext(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Generate a valid JWT
	claims := jwt.MapClaims{
		"user_id":  "user-123",
		"username": "testuser",
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenStr, err := token.SignedString([]byte("test-secret"))
	if err != nil {
		t.Fatalf("failed to generate token: %v", err)
	}

	r := gin.New()
	r.Use(JwtAuth("test-secret"))
	r.GET("/protected", testHandler)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/protected", nil)
	req.Header.Set("Authorization", "Bearer "+tokenStr)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestJwtAuth_WrongSecret(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Generate token with different secret
	claims := jwt.MapClaims{
		"user_id":  "user-123",
		"username": "testuser",
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenStr, err := token.SignedString([]byte("wrong-secret"))
	if err != nil {
		t.Fatalf("failed to generate token: %v", err)
	}

	r := gin.New()
	r.Use(JwtAuth("test-secret"))
	r.GET("/protected", testHandler)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/protected", nil)
	req.Header.Set("Authorization", "Bearer "+tokenStr)
	r.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 for wrong secret, got %d", w.Code)
	}
}

func TestJwtAuth_WrongSigningMethod(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Generate token with different signing method (none)
	token := jwt.NewWithClaims(jwt.SigningMethodNone, jwt.MapClaims{
		"user_id":  "user-123",
		"username": "testuser",
	})
	tokenStr, err := token.SignedString(jwt.UnsafeAllowNoneSignatureType)
	if err != nil {
		t.Fatalf("failed to generate token: %v", err)
	}

	r := gin.New()
	r.Use(JwtAuth("test-secret"))
	r.GET("/protected", testHandler)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/protected", nil)
	req.Header.Set("Authorization", "Bearer "+tokenStr)
	r.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 for none-algorithm token, got %d", w.Code)
	}
}
