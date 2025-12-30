.PHONY: help start stop restart logs status export export-traces export-metadata clean \
        logs-worker logs-web logs-minio check-ports list-exports \
        download-export download-latest-export list-db-exports \
        get-export-urls download-by-id

# Default target
help:
	@echo "Langfuse Management Commands:"
	@echo ""
	@echo "  make start               - Start Langfuse services"
	@echo "  make stop                - Stop Langfuse services"
	@echo "  make restart             - Restart Langfuse services"
	@echo "  make status              - Show status of all services"
	@echo "  make logs                - Show logs from all services"
	@echo "  make logs-worker         - Show logs from worker service"
	@echo "  make logs-web            - Show logs from web service"
	@echo "  make logs-minio          - Show logs from Minio service"
	@echo "  make check-ports         - Check which ports are in use"
	@echo "  make list-exports        - List exported files in Minio"
	@echo "  make list-db-exports     - List exports from database"
	@echo "  make get-export-urls     - Get fresh download URLs for exports"
	@echo "  make download-by-id ID=  - Download export by ID (bypass URL)"
	@echo "  make download-export FILE= - Download specific export file"
	@echo "  make download-latest-export - Download most recent export"
	@echo "  make export              - Export traces via API to JSONL"
	@echo "  make export-traces FILE= - Export traces via API to file"
	@echo "  make export-metadata     - Export trace metadata (sizes only) to JSONL"
	@echo "  make clean               - Remove all data and containers"
	@echo ""

# Start Langfuse
start:
	docker compose up -d

# Stop Langfuse
stop:
	docker compose down

# Restart Langfuse (after config changes)
restart:
	docker compose up -d minio langfuse-web langfuse-worker

# Show status
status:
	docker compose ps

# Check which ports are in use
check-ports:
	@echo "Checking ports..."
	@echo "Port 3000 (Langfuse UI):" $$(lsof -i :3000 2>/dev/null | tail -n +2 || echo "Not in use")
	@echo "Port 9000 (Minio API):" $$(lsof -i :9000 2>/dev/null | tail -n +2 || echo "Not in use")
	@echo "Port 9001 (Minio Console):" $$(lsof -i :9001 2>/dev/null | tail -n +2 || echo "Not in use")

# List exported files via Minio client
list-exports:
	@echo "Listing exports from Minio..."
	@docker exec langfuse-minio mc alias set local http://localhost:9000 minio miniosecret 2>/dev/null || true
	@docker exec langfuse-minio mc ls local/langfuse/exports/ 2>/dev/null || echo "No exports found or Minio not ready"

# List exports from database with download URLs
list-db-exports:
	@echo "Fetching exports from database..."
	@python3 list_exports_db.py

# Get fresh presigned download URLs for all exports
get-export-urls:
	@echo "Generating fresh download URLs..."
	@python3 get_export_urls.py

# Download export by database ID (bypasses presigned URL)
download-by-id:
	@if [ -z "$(ID)" ]; then \
		echo "Usage: make download-by-id ID=<export_id>"; \
		echo ""; \
		echo "Available exports:"; \
		docker exec langfuse-postgres psql -U langfuse -d langfuse -t -c "SELECT id, name FROM batch_exports WHERE status = 'COMPLETED' ORDER BY created_at DESC;"; \
		exit 1; \
	fi
	@python3 download_export.py $(ID)

# Download export file from Minio (usage: make download-export FILE=filename.jsonl)
download-export:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make download-export FILE=filename.jsonl"; \
		echo "Run 'make list-exports' to see available files"; \
		exit 1; \
	fi
	@echo "Downloading $(FILE) from Minio..."
	@docker exec langfuse-minio mc alias set local http://localhost:9000 minio miniosecret 2>/dev/null
	@docker exec langfuse-minio mc cp local/langfuse/exports/$(FILE) - > $(FILE)
	@echo "Downloaded to $(FILE)"

# Download the most recent export
download-latest-export:
	@echo "Finding most recent export..."
	@docker exec langfuse-minio mc alias set local http://localhost:9000 minio miniosecret >/dev/null 2>&1
	@docker exec langfuse-minio mc ls local/langfuse/exports/ 2>/dev/null | tail -1 > .export_file_line.txt
	@if [ ! -s .export_file_line.txt ]; then \
		echo "No exports found"; \
		rm -f .export_file_line.txt; \
		exit 1; \
	fi
	@export_file=$$(cat .export_file_line.txt | tr -s ' ' | cut -d' ' -f6); \
	rm -f .export_file_line.txt; \
	echo "Downloading $$export_file..."; \
	docker exec langfuse-minio sh -c "mc cat local/langfuse/exports/$$export_file" > $$export_file; \
	echo "Downloaded to $$export_file"

# Show all logs
logs:
	docker compose logs -f

# Show worker logs
logs-worker:
	docker compose logs -f langfuse-worker

# Show web logs
logs-web:
	docker compose logs -f langfuse-web

# Show Minio logs
logs-minio:
	docker compose logs -f minio

# Export traces with auto-generated filename
export:
	python3 export_traces.py

# Export traces to specific file
export-traces:
	@if [ -z "$(FILE)" ]; then \
		python3 export_traces.py; \
	else \
		python3 export_traces.py $(FILE); \
	fi

# Export trace metadata only (sizes, no content)
export-metadata:
	python3 export_metadata.py

# Clean everything (including data)
clean:
	docker compose down -v
	@echo "All data and containers removed"
