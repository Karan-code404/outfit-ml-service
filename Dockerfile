FROM python:3.10-slim

# Create user with UID 1000 for Hugging Face Spaces security compliance
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /app

# Copy and install dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download & cache the Hugging Face model files during build to ensure instant Space startup
RUN python -c "\
from transformers import AutoModel, AutoProcessor;\
AutoModel.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True);\
AutoProcessor.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)\
"

# Copy project files
COPY --chown=user:user . .

# Hugging Face Spaces expects service on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
