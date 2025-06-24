FROM python:3.11-slim

WORKDIR /app

RUN pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 8080
CMD ["python", "recommend_backend.py"]
