FROM python:3.12

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["streamlit", "run", "ui.py", "--server.port=3000", "--server.address=0.0.0.0"]