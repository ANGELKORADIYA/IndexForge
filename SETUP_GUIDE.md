# MemorySearch Setup Guide 🔍

This guide will walk you through setting up the MemorySearch environment for development and production.

---

## 1. Prerequisites

Before you begin, ensure you have the following installed:

- **Docker & Docker Compose:** Required for the recommended "Quick Start" and database management.
- **Rust (Optional, for local development):** If you plan to run the code directly on your host machine.
  - Install via [rustup](https://rustup.rs/): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **PostgreSQL & pgvector (Optional, for manual setup):** If you prefer not to use Docker for the database.

---

## 2. Quick Start (Docker - Recommended)

The easiest way to get started is using Docker Compose with Profiles.

### Development Mode
Runs the application with source code mounting and dependency caching.

```bash
# Start the database and the dev container
docker compose --profile dev up
```

### Production Mode
Builds an optimized, minimal binary image.

```bash
# Build and start in production mode
docker compose --profile prod up --build
```

---

## 3. Manual Setup (Local Development)

If you prefer to run the Rust code directly on your machine:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd memory-search
    ```

2.  **Setup the Database:**
    You can use Docker just for the database:
    ```bash
    docker compose up db -d
    ```

3.  **Environment Variables:**
    Copy the example environment file and adjust if necessary:
    ```bash
    cp .env.example .env
    ```

4.  **Install Dependencies:**
    Ensure you have `pkg-config` and `libssl-dev` (on Linux) or `openssl` (on macOS).

5.  **Run the CLI:**
    ```bash
    cargo run --package ms-cli -- --help
    ```

---

## 4. Model Download

MemorySearch requires offline models for embedding and re-ranking.

1.  **Ensure you have the `scripts/download_models.sh` script.** (Create it if it doesn't exist based on PROJECT.md).
2.  **Run the download script:**
    ```bash
    chmod +x scripts/download_models.sh
    ./scripts/download_models.sh
    ```
    This will download approximately 350MB of models into the `./models` directory.

---

## 5. Common Troubleshooting

### `pgvector` extension not found
If you are running PostgreSQL manually, ensure you have installed the `pgvector` extension. 
In SQL: `CREATE EXTENSION IF NOT EXISTS vector;`

### Port 5432 already in use
If you have a local PostgreSQL instance running, it might conflict with the Docker container. 
Change `POSTGRES_PORT` in your `.env` and update the mapping in `docker-compose.yml`.

### Database connection refused
Ensure the `db` service is healthy before the `app` service starts. Docker Compose with `condition: service_healthy` handles this, but manual runs might need a moment for the DB to initialize.

### Slow Rust builds in Docker
The Dockerfile uses a "dummy build" strategy to cache dependencies. However, the first build will always be slow. Subsequent builds that only change your source code will be much faster.

---

## 6. Project Structure

- `crates/`: Modular Rust crates for core logic, ingestion, indexing, etc.
- `data/`: Local storage for Tantivy and other indexes (gitignored).
- `models/`: Offline ML models (gitignored).
- `migrations/`: SQL migration files for database schema.
