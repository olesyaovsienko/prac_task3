FROM python:3.12

WORKDIR /Prac_API

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ensembles /Prac_API/ensembles

CMD ["uvicorn", "ensembles.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

