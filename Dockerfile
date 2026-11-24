# syntax=docker/dockerfile:1
#
# Distributable engine image for gtvdb_core. The Python driver (pip install
# gtvdb) is a client only; this image is the missing server half.
#
# Two binaries, one image:
#   gtv              - the zero-extension REPL (interactive SQL over core crates)
#   gtv-demo-server  - serve_demo, a protocol-v1 gRPC server for client dev
#
#   docker build -t gtvdb-core:0.1.0 .
#   docker run --rm -it gtvdb-core:0.1.0                 # REPL
#   docker run --rm -p 50051:50051 -e GTV_ADDR=0.0.0.0:50051 \
#       gtvdb-core:0.1.0 gtv-demo-server                 # engine for the driver
#
# The demo server holds ONE immutable snapshot for protocol development; the
# production server (gtv-server with a real store) lives in the companion
# gtvdb_ext repo. Do not mistake this for the production engine.

# ---- build stage -----------------------------------------------------------
FROM rust:1.95-slim AS builder
WORKDIR /src

# The base image already IS rust 1.95.0, matching rust-toolchain.toml's pin.
# Do NOT copy rust-toolchain.toml here: that makes rustup re-sync the channel
# and re-download components on every build (slow, non-deterministic). The
# lock file plus --locked is the source of truth inside the image.
# Only the crates tree is needed: no benchmark data, no docs, no generated data.
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
RUN cargo build --release --locked -p gtv-cli --bin gtv && \
    cargo build --release --locked -p gtv-engine --example serve_demo

# ---- runtime stage ---------------------------------------------------------
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/target/release/gtv                        /usr/local/bin/gtv
COPY --from=builder /src/target/release/examples/serve_demo        /usr/local/bin/gtv-demo-server
WORKDIR /data
VOLUME ["/data"]
EXPOSE 50051
# Dispatch so `docker run <image> [gtv|gtv-demo-server]` selects the binary.
# A fixed ENTRYPOINT ["gtv"] would append the arg to gtv instead of replacing it.
ENTRYPOINT ["/bin/sh", "-c", "case \"${1:-gtv}\" in gtv|gtv-demo-server) exec \"$@\" ;; *) echo \"unknown binary: $1 (use gtv or gtv-demo-server)\" >&2; exit 2 ;; esac", "sh"]
