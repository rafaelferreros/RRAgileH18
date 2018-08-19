FROM python:3
ADD dataset dataset
ADD scripts scripts
ADD requirements.txt /

RUN pip install -r requirements.txt

CMD [ "python", "scripts/manu.py" ]