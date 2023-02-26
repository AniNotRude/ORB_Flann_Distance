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
vid2.set(cv2.CAP_PROP_POS_FRAMES, 20)

while vid1.isOpened():

    dummy1, img1 = vid1.read()
    dummy2, img2 = vid2.read()
    #Setting up ORB detector
    orb = cv2.ORB_create()
    key1, des1 = orb.detectAndCompute(img1, None)
    key2, des2 = orb.detectAndCompute(img2, None)

    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    #Setting Hyperparameters for the index parameters
    index_para = dict(algorithm=6, table_number=6, table_size=20,multi_probe_level=2 )
    search_para = {}

    # matcher = cv2.FlannBasedMatcher(index_para,search_para)
    # matches = matcher.match(des1, des2)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(des1, des2, 2)

    #Filter matching with Lowe's ratio
    threshold = 0.7
    good_matches= []
    for i,j in knn_matches:
        if i.distance < j.distance*threshold:
            good_matches.append(i)

    final = cv2.drawMatches(img1, key1, img2, key2, good_matches[:10], None)
    # travel = calc_distance(key1[0], key1[1], key2[0], key2[1])

    for i in range(len(good_matches)):
        # calculate Euclidean distance between keypoints
        distance = calc_distance(key1[good_matches[i].queryIdx].pt[0], key1[good_matches[i].queryIdx].pt[1], key2[good_matches[i].trainIdx].pt[0], key2[good_matches[i].trainIdx].pt[1])
        print("Distance between keypoint {} in frame 1 and keypoint {} in frame 2: {} pixels".format(good_matches[i].queryIdx, good_matches[i].trainIdx, distance))

        

    cv2.imshow("Naam hai ye",final)

    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break
cv2.destroyAllWindows()
vid1.release()

