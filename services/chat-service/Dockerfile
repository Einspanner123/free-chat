FROM golang:1.25-alpine AS builder

WORKDIR /app

# 复制go.mod和go.sum文件
COPY go.mod go.sum ./

# 下载依赖
RUN go mod download

# 复制所有代码
COPY . .

# 设置Go环境变量
ENV CGO_ENABLED=0
ENV GOOS=linux

# 构建应用
RUN go build -o chat-service ./cmd/chat-service/main.go

# 使用Alpine作为运行时基础镜像
FROM alpine:3.20

WORKDIR /app

# 复制构建好的二进制文件
COPY --from=builder /app/chat-service .

# 添加时区支持
RUN apk --no-cache add tzdata

# 运行应用
CMD ["./chat-service"]
