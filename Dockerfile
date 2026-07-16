FROM python:3.10-slim

# Create user with UID 1000 for Hugging Face Spaces security compliance
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /app

# Copy and install dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download & cache the Hugging Face model files during build using OpenCLIP (prevents runtime download and meta tensor bugs)
RUN python -c "\
import open_clip;\
open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')\
"

# Copy project files
COPY --chown=user:user . .

# Hugging Face Spaces/Render expect service on dynamic ports (default 7860/10000)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
