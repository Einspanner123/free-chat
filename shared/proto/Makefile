# Makefile for micro-service proto code generation
# 放置在项目根目录: /home/linkst/workplace/micro-service/Makefile

# 配置项
PROTO_DIR := .
PROTO_FILES := $(wildcard $(PROTO_DIR)/*.proto)
GO_PROTO_OUT := $(PROTO_DIR)
PROTOC := protoc
GO_PLUGIN := protoc-gen-go
GRPC_PLUGIN := protoc-gen-go-grpc

# 检查protoc是否安装
ifeq ($(shell command -v $(PROTOC) 2>/dev/null),)
$(error "protoc编译器未安装，请先安装: sudo apt-get install protobuf-compiler")
endif

# 检查Go插件
check-plugins:
	@echo "检查protoc插件..."
	@which $(GO_PLUGIN) >/dev/null || (echo "需要安装protoc-gen-go..."; \
		go install google.golang.org/protobuf/cmd/protoc-gen-go@latest)
	@which $(GRPC_PLUGIN) >/dev/null || (echo "需要安装protoc-gen-go-grpc..."; \
		go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest)
# export PATH="$PATH:$(go env GOPATH)/bin"

# protoc --go_out=./	--go-grpc_out=./	./proto/summarizer.proto
# 生成Go代码
generate: check-plugins
	@echo "✅ 开始生成proto代码..."
	@echo "   处理目录: $(PROTO_DIR)"
	@echo "   Proto文件: $(PROTO_FILES)"
	@set -e; \
	for proto in $(PROTO_FILES); do \
		dir_name=$$(basename $$proto .proto); \
		mkdir -p $(GO_PROTO_OUT)/$$dir_name; \
		echo "   → 处理 $$proto → 生成到 $(GO_BASE_OUT)/$$dir_name"; \
		$(PROTOC) \
			--go_out=$(GO_PROTO_OUT)/$$dir_name \
			--go_opt=paths=source_relative \
			--go-grpc_out=$(GO_PROTO_OUT)/$$dir_name \
			--go-grpc_opt=paths=source_relative \
			-I $(PROTO_DIR) \
			$$proto; \
	done

# 清理生成的代码
clean:
	@echo "🧹 清理生成的Go代码..."
	@for proto in $(PROTO_FILES); do \
		dir_name=$$(basename $$proto .proto); \
		rm -r $$dir_name; \
		echo "   → $$dir_name is removed"; \
	done
# 	@rm -f $(GO_PROTO_OUT)/*.pb.go
	@echo "清理完成"

# 安装所有依赖
install-deps:
	@echo "⏬ 安装protoc依赖..."
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
	@echo "✅ 依赖安装完成"

# 默认目标
all: generate

.PHONY: generate clean install-deps all check-plugins