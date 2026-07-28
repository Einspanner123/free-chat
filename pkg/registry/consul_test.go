package registry

import "testing"

func TestResolveAdvertiseIP_ReturnsGivenIP(t *testing.T) {
	ip, err := ResolveAdvertiseIP("100.100.1.1")
	if err != nil {
		t.Fatalf("ResolveAdvertiseIP failed: %v", err)
	}
	if ip != "100.100.1.1" {
		t.Errorf("expected 100.100.1.1, got %s", ip)
	}
}
