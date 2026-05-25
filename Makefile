HOST ?= 127.0.0.1
FRONTEND_PORT ?= 5177
BACKEND_PORT ?= 8787

.PHONY: install backend frontend dev build check-data clean

install:
	npm install

backend:
	HOST=$(HOST) PORT=$(BACKEND_PORT) npm run backend

frontend:
	VITE_API_BASE=http://$(HOST):$(BACKEND_PORT) npm run frontend -- --host $(HOST) --port $(FRONTEND_PORT)

dev:
	@backend_pid=; frontend_pid=; \
	trap 'test -n "$$backend_pid" && kill $$backend_pid 2>/dev/null; test -n "$$frontend_pid" && kill $$frontend_pid 2>/dev/null' INT TERM EXIT; \
	HOST=$(HOST) PORT=$(BACKEND_PORT) npm run backend & backend_pid=$$!; \
	VITE_API_BASE=http://$(HOST):$(BACKEND_PORT) npm run frontend -- --host $(HOST) --port $(FRONTEND_PORT) & frontend_pid=$$!; \
	wait $$backend_pid $$frontend_pid

build:
	npm run build

check-data:
	npm run check-data

clean:
	rm -rf dist
