# StyLoop Outfit Recommendation ML Microservice

This is a standalone Python/FastAPI microservice running the **Marqo/marqo-fashionSigLIP** computer vision model to generate similarity recommendations.

## Hugging Face Spaces Deployment Steps

1. **Create a Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
   - Set a name (e.g., `styloop-recommendation-api`).
   - Select **Docker** as the SDK.
   - Choose the **Blank** template (or select Docker directly).
   - Select the Free tier hardware (CPU basic is fine, GPU is faster if available but not required as CPU-optimized torch is included).
   - Set Space visibility to **Public** or **Private** (if Private, ensure you use a Hugging Face token to call it).

2. **Upload Files**:
   - Clone your Hugging Face Space repository or upload the files directly via the browser interface:
     - `app.py`
     - `requirements.txt`
     - `Dockerfile`
   - Once committed, Hugging Face will automatically build and start the container on port `7860`.

3. **Verify the Endpoint**:
   - Once running, your space will be available at `https://<username>-<space-name>.hf.space`.
   - The health check endpoint `GET https://<username>-<space-name>.hf.space/` should return `{"status": "healthy"}`.
   - The main recommendation endpoint is `POST https://<username>-<space-name>.hf.space/recommend` which accepts a multipart image file.

4. **Link to Backend**:
   - Set the `ML_SERVICE_URL` in your backend `.env` file to:
     ```env
     ML_SERVICE_URL=https://<username>-<space-name>.hf.space/recommend
     ```
