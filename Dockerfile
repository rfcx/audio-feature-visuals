FROM continuumio/miniconda3:4.8.2
LABEL maintainer="Rainforest Connection <dev@rfcx.org>"

COPY environment.yml /tmp/environment.yml
RUN conda env update -n myenv --file /tmp/environment.yml

WORKDIR /app
COPY viscli.py .
COPY datavis datavis

ENTRYPOINT ["conda", "run", "-n", "myenv", "python"]
