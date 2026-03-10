# This is an alternative Dockerfile building a Debian environment from scratch.
# The official 'ghcr.io/xschahl/vaquila' image is based on Debian (python:3.11-slim)
# and is usually the recommended way to run vAquila. However, if you need
# absolute control or a custom base image, here is how you can set it up.

FROM python:3.11-slim

# Set environment variables for Python and pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Install any potential OS dependencies here (e.g. curl, git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Add your OS-level packages here
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install vAquila and its Web UI dependencies
# Note: You can also point to a wheel file or a github repository
RUN pip install "vaquila"

# Expose the default Web UI port
EXPOSE 8787

# Use the vaq command as entrypoint
ENTRYPOINT ["vaq"]

# By default, start the web interface binding to 0.0.0.0
CMD ["ui", "--host", "0.0.0.0", "--port", "8787"]
