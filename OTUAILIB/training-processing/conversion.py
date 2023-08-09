import cv2
import os

def convertTxt(ary):
    image_width = 1920
    image_height = 1080
    _, center_x, center_y, width, height = ary
    # the above values are taken in as a string, need to convert to float
    center_x, center_y, width, height = float(center_x), float(center_y), float(width), float(height)
    
    # convert to box coordinates
    x_min = int(center_x * image_width - (width * image_width) / 2) - 660
    y_min = int(center_y * image_height - (height * image_height) / 2) - 240
    x_max = int(center_x * image_width + (width * image_width) / 2) - 660
    y_max = int(center_y * image_height + (height * image_height) / 2) - 240 
    #convert back to relative coordinates
    new_cent_x = ((center_x*image_width)-660)/600
    new_cent_y = ((center_y*image_height)-240)/600
    new_width = width*(1920.0/600)
    new_height = height*(1920.0/600)
    #end
    return [new_cent_x, new_cent_y, new_width, new_height]

directory = '/home/wni1717/dev/OTUMEDAI/VNN/test_dataset/video_4'
newDirectory = '/home/wni1717/dev/OTUMEDAI/VNN/test_dataset/better_video_4'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        filenm = f.split("/")[-1:][0]
        file_end = filenm.split(".")[-1:][0]
        if(file_end == 'txt'):
            nfile = open(os.path.join(newDirectory, filenm), "a")
            filer = open(f, "r")
            for x in filer:
                inp = x.split(" ")
                if(len(inp) == 5):
                    inp[4] = inp[4].split('\n')[0]
                    newVals = convertTxt(inp)
                    newAnnot =inp[0] +  " " + str(newVals[0]) + " " + str(newVals[1]) + " " + str(newVals[2]) + " " + str(newVals[3]) + "\n"
                    nfile.write(newAnnot)
            nfile.close()
        else:
            img = cv2.imread(f)
            img = img[240:840,660:1260]
            cv2.imwrite(os.path.join(newDirectory, filenm),img)