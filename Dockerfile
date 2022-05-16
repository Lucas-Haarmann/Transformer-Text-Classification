# syntax=docker/dockerfile:1
FROM python:3.7.3-alpine
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
RUN apk add --no-cache gcc musl-dev linux-headers
RUN apk --no-cache add lapack libstdc++ && apk --no-cache add --virtual .builddeps g++ gcc gfortran musl-dev lapack-dev && pip install scipy==1.3 && apk del .builddeps && rm -rf /root/.cache
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["flask", "run"]