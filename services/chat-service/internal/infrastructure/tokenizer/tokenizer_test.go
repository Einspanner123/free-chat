package tokenizer

import "testing"

func TestNewTokenizerReturnsCounting(t *testing.T) {
	tk, err := NewTokenizer("gpt-4")
	if err != nil {
		t.Fatalf("NewTokenizer failed: %v", err)
	}

	count := tk.Count("Hello, world!")
	if count <= 0 {
		t.Errorf("expected positive token count, got %d", count)
	}
}

func TestTokenizerDifferentStrings(t *testing.T) {
	tk, err := NewTokenizer("gpt-4")
	if err != nil {
		t.Fatalf("NewTokenizer failed: %v", err)
	}

	short := tk.Count("hi")
	long := tk.Count("Hello world, this is a much longer sentence that should use more tokens")

	if short >= long {
		t.Errorf("short string (%d) should have fewer tokens than long string (%d)", short, long)
	}
}

func TestTokenizerChinese(t *testing.T) {
	tk, err := NewTokenizer("gpt-4")
	if err != nil {
		t.Fatalf("NewTokenizer failed: %v", err)
	}

	// Chinese text typically has ~1-2 tokens per character
	count := tk.Count("你好，请介绍一下人工智能的发展历程")
	if count <= 0 {
		t.Errorf("expected positive token count for Chinese, got %d", count)
	}
}

func TestTokenizerUnsupportedModelFallsBack(t *testing.T) {
	// Use a made-up model name to trigger fallback
	tk, err := NewTokenizer("fake-unknown-model-v42")
	if err != nil {
		t.Fatalf("NewTokenizer with unknown model should not error: %v", err)
	}

	count := tk.Count("test")
	if count <= 0 {
		t.Errorf("expected fallback to produce positive count, got %d", count)
	}
}

func TestNewTokenizerWithQwenModel(t *testing.T) {
	// Qwen models use similar tokenization to cl100k_base
	tk, err := NewTokenizer("Qwen/Qwen3-0.6B")
	if err != nil {
		t.Fatalf("NewTokenizer failed for Qwen model: %v", err)
	}

	count := tk.Count("Hello")
	if count <= 0 {
		t.Errorf("expected positive token count, got %d", count)
	}
}
