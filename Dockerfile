# Image name: automating_equity2
FROM python:3.10.11
WORKDIR /Automating_Equity2
COPY requirements.txt requirements.txt
RUN $(which python) --upgrade pip
RUN $(which python) --requirement requirements.txt
RUN $(which python) -m pip install statannotations --no-dependencies
RUN $(which python) -m spacy download en_core_web_sm
COPY . .
