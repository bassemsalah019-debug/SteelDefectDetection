# Frontend image: build the Vite app, serve it with nginx (which also proxies the API).
# BUILD CONTEXT = webapp/frontend
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . ./
# Empty base -> same-origin requests; nginx proxies /api /auth /media to the backend.
ENV VITE_API_BASE=""
RUN npx vite build

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
