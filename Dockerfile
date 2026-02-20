# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

ARG INSTALL_PLAYWRIGHT=true

# Install build dependencies
# curl_cffi requires: gcc, libcurl4-openssl-dev, libssl-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libcurl4-openssl-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Copy the local rotator_library for editable install
COPY src/rotator_library ./src/rotator_library

# Install dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Optional Playwright support for GitLab trial automation
# Patchright is preferred over vanilla playwright for better anti-detection stealth.
RUN if [ "$INSTALL_PLAYWRIGHT" = "true" ]; then \
      pip install --no-cache-dir --user patchright playwright-stealth; \
    fi

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies for curl_cffi
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

ARG INSTALL_PLAYWRIGHT=true

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY prompts/ ./prompts/

# Create directories for logs and oauth credentials
RUN mkdir -p logs oauth_creds

# Expose the default port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

# Optional browser install for GitLab trial automation.
# - Google Chrome (real browser) for CDP stealth mode — Arkose Labs cannot
#   distinguish a CDP-attached Chrome from a user-launched one.
# - Patchright's bundled Chromium as fallback.
# - Xvfb so Chrome can run "headed" inside the container (no physical display)
#   which avoids Arkose Labs CAPTCHA detection that triggers on --headless.
RUN if [ "$INSTALL_PLAYWRIGHT" = "true" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends xvfb wget gnupg ca-certificates \
      && wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
      && apt-get install -y --no-install-recommends /tmp/chrome.deb \
      && rm -f /tmp/chrome.deb \
      && rm -rf /var/lib/apt/lists/* \
      && python -m patchright install --with-deps chromium; \
    fi

# Default command - runs proxy with the correct PYTHONPATH
CMD ["python", "src/proxy_app/main.py", "--port", "8317"]
