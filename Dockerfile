FROM python:3.12-slim

RUN pip install huggingface_hub transformers datasets numpy statsmodels plotly scikit-learn kaleido beautifulsoup4

COPY . /app
WORKDIR /app