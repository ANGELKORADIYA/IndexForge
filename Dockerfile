# --- Build Stage ---
FROM rust:1.75-slim-bookworm AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Copy the entire workspace
# Note: We rely on .dockerignore to keep the build context small.
# To optimize build times by caching dependencies separately without manually 
# listing crates, consider using 'cargo-chef'.
COPY . .

# Build the binaries in release mode
RUN cargo build --release

# --- Runtime Stage ---
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binaries from builder
COPY --from=builder /usr/src/app/target/release/ms-cli /usr/local/bin/ms-cli
COPY --from=builder /usr/src/app/target/release/ms-server /usr/local/bin/ms-server

# Create data and models directories
RUN mkdir -p /app/data /app/models

# Expose server port
EXPOSE 8080

# Default command
CMD ["ms-cli", "--help"]
