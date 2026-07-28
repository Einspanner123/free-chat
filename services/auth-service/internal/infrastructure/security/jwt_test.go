package security

import (
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

func TestJWTAccessTokenHasShorterExpiryThanRefresh(t *testing.T) {
	// Setup: access 1h, refresh 72h
	svc := NewJWTService("test-secret", 1, 72)

	accessToken, err := svc.GenerateAccessToken("user-1", "testuser")
	if err != nil {
		t.Fatalf("GenerateAccessToken failed: %v", err)
	}

	refreshToken, err := svc.GenerateRefreshToken("user-1", "testuser")
	if err != nil {
		t.Fatalf("GenerateRefreshToken failed: %v", err)
	}

	accessExpiry := accessToken.ExpiresAt
	refreshExpiry := refreshToken.ExpiresAt

	// Access token expiry MUST be before refresh token expiry
	if !accessExpiry.Before(refreshExpiry) {
		t.Errorf("access token expiry (%v) should be before refresh token expiry (%v): "+
			"BUG: generate() always uses refreshExpiration for both token types", accessExpiry, refreshExpiry)
	}

	// Access should be ~1h from now, refresh ~72h from now
	expectedAccessExpiry := time.Now().Add(1 * time.Hour)
	expectedRefreshExpiry := time.Now().Add(72 * time.Hour)

	slop := 1 * time.Minute
	if accessExpiry.Before(expectedAccessExpiry.Add(-slop)) || accessExpiry.After(expectedAccessExpiry.Add(slop)) {
		t.Errorf("access token expiry %v should be near %v (±%v)", accessExpiry, expectedAccessExpiry, slop)
	}
	if refreshExpiry.Before(expectedRefreshExpiry.Add(-slop)) || refreshExpiry.After(expectedRefreshExpiry.Add(slop)) {
		t.Errorf("refresh token expiry %v should be near %v (±%v)", refreshExpiry, expectedRefreshExpiry, slop)
	}
}

func TestJWTGenerateAccessTokenClaims(t *testing.T) {
	svc := NewJWTService("test-secret", 1, 72)

	token, err := svc.GenerateAccessToken("user-1", "testuser")
	if err != nil {
		t.Fatalf("GenerateAccessToken failed: %v", err)
	}
	if token.Token == "" {
		t.Error("access token should not be empty")
	}

	claims := &Claims{}
	parsed, err := jwt.ParseWithClaims(token.Token, claims, func(token *jwt.Token) (any, error) {
		return []byte("test-secret"), nil
	})
	if err != nil {
		t.Fatalf("failed to parse token: %v", err)
	}
	if !parsed.Valid {
		t.Error("token should be valid")
	}
	if claims.UserID != "user-1" {
		t.Errorf("expected user_id 'user-1', got '%s'", claims.UserID)
	}
	if claims.Username != "testuser" {
		t.Errorf("expected username 'testuser', got '%s'", claims.Username)
	}
	if claims.Subject != TypeAccess {
		t.Errorf("expected subject '%s', got '%s'", TypeAccess, claims.Subject)
	}
}

func TestJWTGenerateRefreshTokenClaims(t *testing.T) {
	svc := NewJWTService("test-secret", 1, 72)

	token, err := svc.GenerateRefreshToken("user-1", "testuser")
	if err != nil {
		t.Fatalf("GenerateRefreshToken failed: %v", err)
	}
	if token.Token == "" {
		t.Error("refresh token should not be empty")
	}

	claims := &Claims{}
	parsed, err := jwt.ParseWithClaims(token.Token, claims, func(token *jwt.Token) (any, error) {
		return []byte("test-secret"), nil
	})
	if err != nil {
		t.Fatalf("failed to parse token: %v", err)
	}
	if !parsed.Valid {
		t.Error("token should be valid")
	}
	if claims.UserID != "user-1" {
		t.Errorf("expected user_id 'user-1', got '%s'", claims.UserID)
	}
	if claims.Username != "testuser" {
		t.Errorf("expected username 'testuser', got '%s'", claims.Username)
	}
	if claims.Subject != TypeRefresh {
		t.Errorf("expected subject '%s', got '%s'", TypeRefresh, claims.Subject)
	}
}

func TestJWTValidateTokenValid(t *testing.T) {
	svc := NewJWTService("test-secret", 1, 72)

	token, err := svc.GenerateAccessToken("user-1", "testuser")
	if err != nil {
		t.Fatalf("GenerateAccessToken failed: %v", err)
	}

	valid, err := svc.ValidateToken(token.Token)
	if err != nil {
		t.Fatalf("ValidateToken failed: %v", err)
	}
	if !valid {
		t.Error("token should be valid")
	}
}

func TestJWTValidateTokenInvalid(t *testing.T) {
	svc := NewJWTService("test-secret", 1, 72)

	valid, err := svc.ValidateToken("invalid.token.here")
	if err == nil {
		t.Error("should return error for invalid token")
	}
	if valid {
		t.Error("token should not be valid")
	}
}

func TestJWTRefreshTokenSuccess(t *testing.T) {
	svc := NewJWTService("test-secret", 1, 72)

	refreshToken, err := svc.GenerateRefreshToken("user-1", "testuser")
	if err != nil {
		t.Fatalf("GenerateRefreshToken failed: %v", err)
	}

	accessToken, newRefreshToken, err := svc.RefreshToken(refreshToken.Token)
	if err != nil {
		t.Fatalf("RefreshToken failed: %v", err)
	}

	if accessToken.Token == "" {
		t.Error("new access token should not be empty")
	}
	if newRefreshToken.Token == "" {
		t.Error("new refresh token should not be empty")
	}

	valid, err := svc.ValidateToken(accessToken.Token)
	if err != nil || !valid {
		t.Error("new access token should be valid")
	}
}

func TestJWTRefreshTokenInvalid(t *testing.T) {
	svc := NewJWTService("test-secret", 1, 72)

	_, _, err := svc.RefreshToken("invalid.token.here")
	if err == nil {
		t.Error("should return error for invalid refresh token")
	}
}

func TestJWTRefreshTokenWithAccessTokenFails(t *testing.T) {
	svc := NewJWTService("test-secret", 1, 72)

	accessToken, err := svc.GenerateAccessToken("user-1", "testuser")
	if err != nil {
		t.Fatalf("GenerateAccessToken failed: %v", err)
	}

	// Using an access token (subject=access) as a refresh token should fail
	_, _, err = svc.RefreshToken(accessToken.Token)
	if err == nil {
		t.Error("should return error when using access token for refresh")
	}
}
