FROM tensorflow/tensorflow:1.15.2-py3
RUN apt-get update && apt-get install -yq libgl1-mesa-glx git
COPY ./* /gcp_tutorial/
RUN pip install -r /gcp_tutorial/requirements.txt
RUN mkdir /log
WORKDIR /gcp_tutorial
