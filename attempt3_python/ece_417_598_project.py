#!/usr/bin/python3
import cv2
import numpy as np

# Dimensions of checkerboard
checkerboard_dims = (7, 9)
critera = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.001)

obj_points = []
img_points = []

objp = np.zeros((1, checkerboard_dims[0] * checkerboard_dims[1], 3), \
        np.float32)
objp[0,:,:2] = np.mgrid[0:checkerboard_dims[0], \
        0:checkerboard_dims[1]].T.reshape(-1, 2)
prev_img_shape = None

vid = cv2.VideoCapture(0)

while (True):
    ret, frame = vid.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(frame, checkerboard_dims, \
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + \
            cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret == True:
        print("ret == True")
        
        obj_points.append(objp)
        
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        img_points.append(corners2)

        frame = cv2.drawChessboardCorners(frame, checkerboard_dims, corners2, ret)
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    h, w = frame.shape[:2]
    
    '''
    ret, mtx, dist, rvecs, tvecs = \
            cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], \
                None, None)

    print(mtx, dist, rvecs, tvecs)
    '''

vid.release()
cv2.destroyAllWindows()

