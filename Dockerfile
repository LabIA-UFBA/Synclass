FROM python:3.11

ARG HOME='/home/ds/'
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    make build-essential zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev \
    libgl1-mesa-glx r-base libssl3


WORKDIR ${HOME}

# copy file with packages requeriments
COPY requirements.txt ${HOME}requirements.txt

# upgrade pip
RUN pip  install --upgrade pip 

## install python packages
RUN pip install -r requirements.txt

# remove requeriments.txt
RUN rm requirements.txt

CMD ["python3"]
