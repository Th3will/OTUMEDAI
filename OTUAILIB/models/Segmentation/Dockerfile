# docker build /home/wni1717/dev/OTUMEDAI/OTUAILIB/models/Segmentation -t test-seg
# docker run --runtime=nvidia -it -p 8888:8888 test-seg
FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN rm -rf tensorflow-tutorials
COPY . .
ENV SM_FRAMEWORK=tf.keras
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
