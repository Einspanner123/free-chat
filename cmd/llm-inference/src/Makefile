# Makefile for Python proto code generation
# 放置在llm-inference/proto目录

# 配置项
PROTO_DIR := .
PROTO_FILES := $(wildcard $(PROTO_DIR)/*.proto)
PY_PROTO_OUT := $(PROTO_DIR)
PROTOC := protoc
UV := uv

# 检查protoc是否安装
ifeq ($(shell command -v $(PROTOC) 2>/dev/null),)
$(error "protoc编译器未安装，请先安装: sudo apt-get install protobuf-compiler")
endif

# 检查uv是否安装
ifeq ($(shell command -v $(UV) 2>/dev/null),)
$(error "uv包管理器未安装，请先安装: curl -LsSf https://astral.sh/uv/install.sh | sh")
endif

# 默认目标
all: generate

# 检查Python依赖
check-deps:
	@echo "检查Python依赖..."
	@$(UV) add --quiet protobuf grpcio grpcio-tools || echo "依赖已安装"
	@echo "✅ Python依赖检查完成"

# 生成Python代码

generate: check-deps
	@echo "✅ 开始生成Python proto代码..."
	@echo "   处理目录: $(PROTO_DIR)"
	@echo "   Proto文件: $(PROTO_FILES)"
	@set -e; \
	for proto in $(PROTO_FILES); do \
		dir_name=$$(basename $$proto .proto); \
# 		mkdir -p $(PY_PROTO_OUT)/$$dir_name; \
		echo "   → 处理 $$proto → 生成到 $(PY_PROTO_OUT)"; \
		$(UV) run python -m grpc_tools.protoc \
			--python_out=$(PY_PROTO_OUT) \
			--grpc_python_out=$(PY_PROTO_OUT) \
			-I $(PROTO_DIR) \
			$$proto; \
		echo "   ✅ $$proto 生成完成"; \
	done


# 清理生成的代码
clean:
	@echo "🧹 清理生成的Python代码..."
	@find $(PY_PROTO_OUT) -name "*.py" -type f -not -name "__init__.py" -delete 2>/dev/null || true
	@echo "清理完成"

# 安装所有依赖
install-deps:
	@echo "⏬ 安装Python依赖..."
	@$(UV) venv --system-site-packages || echo "虚拟环境已存在"
	@$(UV) pip install protobuf grpcio grpcio-tools
	@echo "✅ 依赖安装完成"

.PHONY: generate clean install-deps all check-deps