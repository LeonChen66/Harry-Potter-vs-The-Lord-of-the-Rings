import cv2
vidcap = cv2.VideoCapture('500 Days Of Summer.2009.720p.BDRip.x264-VLiS.mp4')
success, image = vidcap.read()
count = 300 # Per second
while success:
    #if count>60*30:
    cv2.imwrite("test/frame%d.jpg" % count, image)     # save frame as JPEG file
    # Current position of the video file in milliseconds or video capture timestamp.
    vidcap.set(cv2.CAP_PROP_POS_MSEC, (count*1000))
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
