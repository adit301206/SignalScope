FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-deploy.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-deploy.txt

COPY app ./app
COPY src ./src
COPY model ./model

EXPOSE 8000

CMD ["uvicorn", "app.backend:app", "--host", "0.0.0.0", "--port", "8000"]
