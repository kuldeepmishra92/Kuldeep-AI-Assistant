FROM python:3.10-slim

# Create user with UID 1000 to comply with Hugging Face Spaces requirements
RUN useradd -m -u 1000 user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production

# Set the working directory
WORKDIR $HOME/app

# Install system dependencies required for building some packages and sqlite
RUN apt-get update && apt-get install -y \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=user . .

# Make sure directories where the app writes data are writable and owned by the user
RUN mkdir -p $HOME/app/data $HOME/app/logs $HOME/app/memory $HOME/app/tmp && \
    chown -R user:user $HOME/app && \
    chmod -R 777 $HOME/app/data $HOME/app/logs $HOME/app/memory $HOME/app/tmp

# Switch to the non-root user
USER user

# Expose the port Hugging Face Spaces uses
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
