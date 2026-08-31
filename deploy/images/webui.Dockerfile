FROM node:22-bookworm-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5 AS build
WORKDIR /src
COPY webui/package.json webui/package-lock.json ./
RUN npm ci --ignore-scripts
COPY webui ./
RUN npm run build

FROM nginxinc/nginx-unprivileged:mainline-alpine@sha256:d9083fe47768377ef55dedafd67d4da7c2f2bc2bece7554954f29359deb0dce9
COPY deploy/images/webui.nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/dist /usr/share/nginx/html
EXPOSE 8080
