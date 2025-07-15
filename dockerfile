# Use the official Python 3.11.4 slim image as the base for the build stage
FROM python:3.11.4-slim AS builder

# Copy the requirements file into the image
COPY requirements.txt /tmp/requirements.txt

# Install the Python dependencies from requirements.txt into /install
# --no-cache-dir avoids caching to keep the image small
# --target specifies a custom install location
RUN pip install --no-cache-dir --prefix=/usr/local -r /tmp/requirements.txt


# Start a new, clean image for the final stage to keep it lightweight
FROM python:3.11.4-slim

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the final image
WORKDIR /app

# Copy the previously installed dependencies from the builder stage
COPY --from=builder /usr/local /usr/local

# Copy the application code into the final image
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Specify the command to start the streamlit application
CMD ["streamlit", "run", "start.py"]