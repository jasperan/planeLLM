.PHONY: build run test fmt clean

build:
	go build -o bin/planellm-tui ./cmd/planellm-tui

run:
	go run ./cmd/planellm-tui

test:
	go test ./... -v

fmt:
	go fmt ./...

clean:
	rm -rf bin/
