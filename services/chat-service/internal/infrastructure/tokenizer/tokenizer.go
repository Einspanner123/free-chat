package tokenizer

import (
	"strings"
	"sync"

	"github.com/pkoukk/tiktoken-go"
)

var (
	encodingCache sync.Map
)

type Tokenizer struct {
	encoding string
}

var modelToEncoding = map[string]string{
	"gpt-4":            "cl100k_base",
	"gpt-3.5-turbo":    "cl100k_base",
	"text-embedding-3": "cl100k_base",
	"qwen":             "cl100k_base",
}

func resolveEncoding(modelName string) string {
	lower := strings.ToLower(modelName)
	if enc, ok := modelToEncoding[lower]; ok {
		return enc
	}
	if strings.Contains(lower, "qwen") {
		return "cl100k_base"
	}
	if strings.Contains(lower, "gpt") {
		return "cl100k_base"
	}
	return ""
}

func NewTokenizer(modelName string) (*Tokenizer, error) {
	enc := resolveEncoding(modelName)
	return &Tokenizer{encoding: enc}, nil
}

func (t *Tokenizer) Count(text string) int {
	if t.encoding == "" {
		return max(len(text)/2, 1)
	}

	tke, err := t.getEncoding()
	if err != nil {
		return max(len(text)/2, 1)
	}

	tokens := tke.Encode(text, nil, nil)
	return len(tokens)
}

func (t *Tokenizer) getEncoding() (*tiktoken.Tiktoken, error) {
	if cached, ok := encodingCache.Load(t.encoding); ok {
		return cached.(*tiktoken.Tiktoken), nil
	}

	tke, err := tiktoken.GetEncoding(t.encoding)
	if err != nil {
		return nil, err
	}

	encodingCache.Store(t.encoding, tke)
	return tke, nil
}
