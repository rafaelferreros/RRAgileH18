FROM python:3
ADD dataset dataset
ADD scripts scripts
ADD requirements.txt /

RUN pip install -r requirements.txt

VOLUME ["/result"]
CMD [ "python", "scripts/manu.py" ]