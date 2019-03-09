import cv2
import os
import re
"""
Unit start_time & end_tim (min)
     freq (sec)
"""
def movieClipper(movie_name, dir_name, start_time=0, end_time=200,freq=1,resize=False,harry=False):
    file_name = movie_name
    vidcap = cv2.VideoCapture(file_name)
    success, image = vidcap.read()
    count = start_time*1000*60 # Per second
    while success and count < end_time*1000*60:
        #if count>60*30:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count))
        success, image = vidcap.read()
        at_time = count/1000
        moment = "{:.0f}_{:02.0f}".format(at_time//60,at_time%60)
        print("{}/frame_{}_{}.jpg".format(dir_name, movie_name[:-4], moment))
        # resiave image
        if resize==True:
            image = cv2.resize(image,(64,64))

        save_name = "{}/frame{}_{}_{}.jpg".format(
            dir_name, movie_name.split('/')[-2], movie_name.split('/')[-1][:-4], moment)
        if success:
            if harry:
                (h,w,_) = image.shape
                cv2.imwrite( save_name,
                        image[2*h//17:15*h//17,:])     # save frame as JPEG file
            else:
                cv2.imwrite(save_name,image)
        # Current position of the video file in milliseconds or video capture timestamp.
        count += freq*1000
    
    vidcap.release()


def main():
    for(root, dirs, files) in os.walk('Movies/LoR', topdown=True):
        for file in files:
            if re.search(".mkv", file):
                movieClipper(
                    root+'/'+file, 'train/LoR', 3, 168, 10, False,harry=False)

if __name__ == "__main__":
    main()
