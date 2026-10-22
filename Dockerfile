FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-liberation \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces user 1000 ke naam se chalate hain
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    REQUIRE_PASSWORD=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
EXPOSE 10000
# Render PORT env deta hai (default 10000); local par 7860
CMD python -m influencer serve --host 0.0.0.0
