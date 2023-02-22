import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

#Find distance
def calc_distance(x1, y1, x2, y2):
    distance= math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

#Query Image
path1="/home/aninotrude/Downloads/Test1.mp4"
vid1=cv2.VideoCapture(path1)

#Training Image
vid2=cv2.VideoCapture(path1)

#Moving Image-2 20 frames ahead
vid2.set(cv2.CAP_PROP_POS_FRAMES, 50)

while vid2.isOpened():

    dummy1, img1 = vid1.read()
    dummy2, img2 = vid2.read()
    #Setting up ORB detector
    orb = cv2.ORB_create()
    key1, des1 = orb.detectAndCompute(img1, None)
    key2, des2 = orb.detectAndCompute(img2, None)

    #Setting Hyperparameters for the index parameters
    index_para = dict(algorithm=6, table_number=6, table_size=20,multi_probe_level=2 )
    search_para = {}

    matcher = cv2.FlannBasedMatcher(index_para,search_para)
    matches = matcher.match(des1, des2)

    final = cv2.drawMatches(img1, key1, img2, key2, matches[:30], None)
    # travel = calc_distance(key1[0], key1[1], key2[0], key2[1])

    for i in range(len(matches)):
        # calculate Euclidean distance between keypoints
        distance = calc_distance(key1[matches[i].queryIdx].pt[0], key1[matches[i].queryIdx].pt[1], key2[matches[i].trainIdx].pt[0], key2[matches[i].trainIdx].pt[1])
        print("Distance between keypoint {} in frame 1 and keypoint {} in frame 2: {} pixels".format(matches[i].queryIdx, matches[i].trainIdx, distance))

        

    cv2.imshow("Naam hai ye",final)

    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break
cv2.destroyAllWindows()
vid1.release()

