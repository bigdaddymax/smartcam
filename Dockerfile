FROM python:3.7-slim

RUN apt-get -y update
# for dlib
RUN apt-get install -y build-essential cmake
# for opencv
RUN apt-get install -y libopencv-dev

# pip instlal
RUN pip install dlib==19.17.0 \
  && pip install opencv-python==4.1.0.25 \
  && pip install zmq \
  && pip install psutil \
  && pip install werkzeug

COPY . .

EXPOSE 4000

ENTRYPOINT ["python", "server2.py"] 
