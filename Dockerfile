FROM apache/spark:3.5.0

USER root

RUN python3 -m pip install --no-cache-dir graphframes pandas matplotlib pmdarima

USER spark