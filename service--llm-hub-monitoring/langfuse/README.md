# Self-Hosted Langfuse

This directory contains the configuration for running a self-hosted Langfuse instance.

## Quick Start

### 1. Start Langfuse

```bash
cd langfuse
docker compose up -d
```

This will start:
- **Langfuse UI** at http://localhost:3000
- **PostgreSQL database** for storing traces

### 2. Create API Keys

1. Open http://localhost:3000 in your browser
2. Create an account (first time setup)
3. Go to **Settings** → **API Keys**
4. Create a new access key
5. Copy the **Public Key** (starts with `pk-lf-`)
6. Copy the **Secret Key** (starts with `sk-lf-`)

### 3. Configure llm-hub-monitoring

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxx
LANGFUSE_ENABLED=true
LANGFUSE_CAPTURE_CONTENT=true
```

### 4. Run llm-hub-monitoring with Langfuse

```bash
cd ..
# Load environment variables from langfuse/.env
export $(cat langfuse/.env | xargs)
uv run python playground_test_langfuse.py
```

Or with a single command:

```bash
env $(cat langfuse/.env | xargs) uv run python playground_test_langfuse.py
```

## Management

### Check Langfuse status

```bash
docker compose ps
```

### View logs

```bash
# All logs
docker compose logs -f

# Langfuse only
docker compose logs -f langfuse

# PostgreSQL only
docker compose logs -f postgres
```

### Stop Langfuse

```bash
docker compose down
```

### Delete all data (including database)

```bash
docker compose down -v
```

## Updating Langfuse

To update to the latest version:

```bash
docker compose pull
docker compose up -d
```

## Production Usage

For production deployment:

1. **Change passwords**: Update the PostgreSQL password in `compose.yaml`
2. **Generate secure secrets**: Update `NEXTAUTH_SECRET` and `SALT` environment variables
3. **Use external database**: Consider using a managed PostgreSQL service
4. **Set up reverse proxy**: Use nginx or similar for HTTPS
5. **Configure backup**: Set up automated database backups

### Example production compose.yaml changes:

```yaml
environment:
  - DATABASE_URL=postgresql://user:secure_password@external-db:5432/langfuse
  - NEXTAUTH_SECRET=<generate with: openssl rand -base64 32>
  - SALT=<generate with: openssl rand -base64 32>
  - HOST=https://langfuse.yourdomain.com
```

## Troubleshooting

### Langfuse UI not accessible

Check if the container is running:
```bash
docker compose ps
```

Check logs:
```bash
docker compose logs langfuse
```

### Database connection errors

Ensure PostgreSQL is healthy:
```bash
docker compose logs postgres
```

### Cannot create API keys

- Make sure you've created an account in the UI
- Go to Settings → API Keys
- Click "Create Access Key"

## Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [Langfuse GitHub](https://github.com/langfuse/langfuse)
- [Docker Compose Reference](https://docs.docker.com/compose/)
