FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY models.py .
COPY server/ ./server/
COPY openenv.yaml .

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Run Gradio dashboard on port 7860
CMD ["python", "server/gradio_app.py"]
