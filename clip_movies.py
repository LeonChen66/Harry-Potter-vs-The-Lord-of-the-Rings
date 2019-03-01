import cv2

"""
Unit start_time & end_tim (min)
     freq (sec)
"""
def movieClipper(movie_name, dir_name, start_time=0, end_time=200,freq=1):
    file_name = 'Movies/'+movie_name
    vidcap = cv2.VideoCapture(file_name)
    success, image = vidcap.read()
    count = start_time*1000*60 # Per second
    while success and count < end_time*1000*60:
        #if count>60*30:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count))
        success, image = vidcap.read()
        at_time = count/1000
        moment = "{:.0f}_{:02.0f}".format(at_time//60,at_time%60)
        #print(moment)
        cv2.imwrite("{}/frame{}.jpg".format(dir_name, moment),
                    image)     # save frame as JPEG file
        # Current position of the video file in milliseconds or video capture timestamp.

        print('Read a new frame: ', success)
        count += freq*1000

def main():
    movieClipper('500 Days Of Summer.2009.720p.BDRip.x264-VLiS.mp4','Got',10,180,60)

if __name__ == "__main__":
    main()
