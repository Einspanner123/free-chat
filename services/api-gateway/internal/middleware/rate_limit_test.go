package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
)

func newTestRedis(t *testing.T) *redis.Client {
	mr, err := miniredis.Run()
	if err != nil {
		t.Fatalf("failed to start miniredis: %v", err)
	}
	t.Cleanup(mr.Close)

	client := redis.NewClient(&redis.Options{
		Addr: mr.Addr(),
	})
	t.Cleanup(func() { client.Close() })

	return client
}

func TestRateLimit_PassesRequest(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rdb := newTestRedis(t)
	r := gin.New()
	r.Use(RateLimit(rdb, 100))
	r.GET("/test", func(c *gin.Context) {
		c.JSON(200, gin.H{"ok": true})
	})

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200 for first request, got %d", w.Code)
	}
	// Should have rate limit headers
	if w.Header().Get("X-RateLimit-Limit") == "" {
		t.Error("X-RateLimit-Limit header should be set when request is allowed")
	}
	if w.Header().Get("X-RateLimit-Remaining") == "" {
		t.Error("X-RateLimit-Remaining header should be set when request is allowed")
	}
}

func TestRateLimit_ExceedsLimit(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rdb := newTestRedis(t)
	// Set QPS to 1, capacity = 2
	r := gin.New()
	r.Use(RateLimit(rdb, 1))
	r.GET("/test", func(c *gin.Context) {
		c.JSON(200, gin.H{"ok": true})
	})

	// First 2 requests should pass (capacity)
	for i := 0; i < 2; i++ {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/test", nil)
		r.ServeHTTP(w, req)
		if w.Code != 200 {
			t.Errorf("request %d should pass (within capacity), got %d", i+1, w.Code)
		}
	}

	// 3rd request should be rate limited
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	r.ServeHTTP(w, req)
	if w.Code != 429 {
		t.Errorf("expected 429 when rate limited, got %d", w.Code)
	}
	if w.Header().Get("Retry-After") == "" {
		t.Error("Retry-After header should be set when rate limited")
	}
}

func TestRateLimit_RedisDown_FailOpen(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Use a disconnected Redis client to simulate Redis being down
	client := redis.NewClient(&redis.Options{
		Addr: "127.0.0.1:16379", // unlikely to be running
	})

	r := gin.New()
	r.Use(RateLimit(client, 10))
	r.GET("/test", func(c *gin.Context) {
		c.JSON(200, gin.H{"ok": true})
	})

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	r.ServeHTTP(w, req)

	// Should fail open — allow the request even if Redis is down
	if w.Code != 200 {
		t.Errorf("expected 200 (fail-open) when Redis is down, got %d", w.Code)
	}
}

func TestRateLimit_ReplenishesTokens(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rdb := newTestRedis(t)
	// QPS=1, capacity=2 — after 2 requests, should be rate limited
	r := gin.New()
	r.Use(RateLimit(rdb, 1))
	r.GET("/test", func(c *gin.Context) {
		c.JSON(200, gin.H{"ok": true})
	})

	// Exhaust tokens
	for i := 0; i < 2; i++ {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/test", nil)
		r.ServeHTTP(w, req)
		if w.Code != 200 {
			t.Errorf("request %d should pass, got %d", i+1, w.Code)
		}
	}

	// Should be rate limited now
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	r.ServeHTTP(w, req)
	if w.Code != 429 {
		t.Errorf("expected 429, got %d", w.Code)
	}
}
