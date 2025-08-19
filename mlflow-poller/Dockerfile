FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .

EXPOSE 8080

ENV PYTHONUNBUFFERED 1
CMD ["python","-u", "main.py"]