# docker build /home/wni1717/dev/OTUMEDAI/OTUAILIB/models/Segmentation -t test-seg
# docker run --runtime=nvidia -it -p 8888:8888 test-seg
FROM python:3.9-slim
WORKDIR /code
COPY ./Polyp_Segmentation_Task /code/Polyp_Segmentation_Task
COPY ./requirements.txt /code/requirements.txt
COPY ./training /code/training
ENV SM_FRAMEWORK=tf.keras
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN cd /code
CMD [ "python", "Polyp_Segmentation_Task/full_train.py" ]