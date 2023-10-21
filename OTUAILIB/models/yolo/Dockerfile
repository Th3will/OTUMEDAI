# sudo docker build /home/wni1717/dev/OTUMEDAI/OTUAILIB/models/yolo/ -t yolo
# sudo create yolo
# sudo run <name> -it --ipc=host --gpus all $t
FROM ultralytics/ultralytics:latest
WORKDIR /code
COPY . .
RUN ls -al
RUN pip install -r requirements.txt

