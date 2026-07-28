package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestCORS_HEAD_Request(t *testing.T) {
	// RED: test will verify CORS headers are set
	gin.SetMode(gin.TestMode)

	r := gin.New()
	r.Use(CORS())
	r.GET("/test", func(c *gin.Context) {
		c.JSON(200, gin.H{"ok": true})
	})

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	r.ServeHTTP(w, req)

	if w.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Errorf("expected Access-Control-Allow-Origin: *, got: %s", w.Header().Get("Access-Control-Allow-Origin"))
	}
	if w.Header().Get("Access-Control-Allow-Methods") == "" {
		t.Error("Access-Control-Allow-Methods header should be set")
	}
	if w.Header().Get("Access-Control-Allow-Headers") == "" {
		t.Error("Access-Control-Allow-Headers header should be set")
	}
	if w.Header().Get("Access-Control-Max-Age") != "86400" {
		t.Errorf("expected Access-Control-Max-Age: 86400, got: %s", w.Header().Get("Access-Control-Max-Age"))
	}
	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestCORS_OPTIONS_Returns204(t *testing.T) {
	gin.SetMode(gin.TestMode)

	r := gin.New()
	r.Use(CORS())
	r.POST("/test", func(c *gin.Context) {
		c.JSON(200, gin.H{"ok": true})
	})

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("OPTIONS", "/test", nil)
	r.ServeHTTP(w, req)

	// OPTIONS should return 204 and NOT call the handler
	if w.Code != 204 {
		t.Errorf("expected 204 for OPTIONS, got %d", w.Code)
	}
	// CORS headers should still be present
	if w.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Errorf("expected Access-Control-Allow-Origin: *, got: %s", w.Header().Get("Access-Control-Allow-Origin"))
	}
}

func TestCORS_OPTIONS_HandlerNotCalled(t *testing.T) {
	gin.SetMode(gin.TestMode)

	called := false
	r := gin.New()
	r.Use(CORS())
	r.GET("/test", func(c *gin.Context) {
		called = true
		c.JSON(200, gin.H{"ok": true})
	})

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("OPTIONS", "/test", nil)
	r.ServeHTTP(w, req)

	if called {
		t.Error("handler should NOT be called for OPTIONS request")
	}
}

func TestCORS_NonOPTIONS_PassesHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	called := false
	r := gin.New()
	r.Use(CORS())
	r.GET("/test", func(c *gin.Context) {
		called = true
		c.JSON(200, gin.H{"ok": true})
	})

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	r.ServeHTTP(w, req)

	if !called {
		t.Error("handler should be called for non-OPTIONS request")
	}
	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
}
