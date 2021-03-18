from os.path import isfile, join
import numpy as np
import cv2 as cv
from cv2 import ximgproc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.fromnumeric import ndim
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


'''Global Variables '''

PATH_L = r'L_stream'
PATH_R = r'R_stream'

### Stereo Matcher Parameters
minDisp = 0     # window position x-offset
nDisp = 96      # Range of visible depths, larger num takes longer (./16)
bSize = 7      # Size of search windows, (typ. 7-15, odd # only)
# P1 = 8*3*bSize**2
# P2 = 32*3*bSize**2
# mode = cv.STEREO_SGBM_MODE_HH4
pfCap = 0
sRange = 0
y_floor = 370      # y-axis pixel location of floor plane (scene-specific)

### Weighted least squares parameters
lam = 64000    # Regularization param
sigma = 2.5      # Contrast sensitivity param
discontinuityRad = 2

params = [minDisp, nDisp, bSize, pfCap, sRange]

### Load Camera Calibration Parameters
undistL = np.loadtxt(r'Calibration_Files\umapL.txt', dtype=np.float32)
rectifL = np.loadtxt(r'Calibration_Files\rmapL.txt', dtype=np.float32)
undistR = np.loadtxt(r'Calibration_Files\umapR.txt', dtype=np.float32)
rectifR = np.loadtxt(r'Calibration_Files\rmapR.txt', dtype=np.float32)
roiL = np.loadtxt(r'Calibration_Files\ROIL.txt', dtype=np.int)
roiR = np.loadtxt(r'Calibration_Files\ROIR.txt', dtype=np.int)
Q = np.loadtxt(r'Calibration_Files\Q.txt')
R = np.loadtxt(r'Calibration_Files\Rtn.txt', dtype=np.float32)
T = np.loadtxt(r'Calibration_Files\Trnsl.txt', dtype=np.float32)
CL = np.loadtxt(r'Calibration_Files\CmL.txt', dtype=np.float32)
DL = np.loadtxt(r'Calibration_Files\DcL.txt', dtype=np.float32)

''' End Global Variables '''


def main():
    useStream = True
    streamFrames = range(40, 100)
    imgPairId = '65.jpg'

    if not (isfile(join(PATH_L, imgPairId)) and isfile(join(PATH_R, imgPairId))):
        print('Image', imgPairId, 'not found')
        exit()

    if (useStream):
        ''' Loop & update figures through image stream ''' 
        plt.figure(figsize=(12, 8))
        for frameId in streamFrames:
            fname = str(frameId) + '.jpg'
            compute_disparity(fname, params)
            plt.pause(0.2)
    else:
        plt.figure(figsize=(12, 8))
        compute_disparity(imgPairId, params)
        plt.show()


def rescaleROI(img, roi):
    x, y, w, h = roi
    dst = img[y:y+h, x:x+w]
    return dst


def compute_disparity(imgId, params):
    imgL = cv.imread(join(PATH_L, imgId))
    imgIdR = int(imgId.split('.')[0]) + 1       # Offset 1-frame delay from L -> R
    imgIdR = str(imgIdR) + '.jpg'
    imgR = cv.imread(join(PATH_R, imgId))

    imgL = cv.remap(imgL, undistL, rectifL, cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT)
    imgR = cv.remap(imgR, undistR, rectifR, cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT)

    imgL = rescaleROI(imgL, roiL)
    imgR = rescaleROI(imgR, roiR)

    dsize = (imgL.shape[1], imgL.shape[0])
    imgR = cv.resize(imgR, dsize, interpolation=cv.INTER_LANCZOS4)

    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    ### Init StereoMatcher with parameters
    (minDisp,nDisp,bSize,pfCap,sRange) = params
    stereoL = cv.StereoSGBM_create(
        minDisparity=minDisp,
        numDisparities=nDisp,
        blockSize=bSize,
        preFilterCap=pfCap,
        speckleRange=sRange)

    ### Init WLS Filter with parameters
    wls = ximgproc.createDisparityWLSFilter(stereoL)
    stereoR = ximgproc.createRightMatcher(stereoL)
    wls.setLambda(lam)
    wls.setDepthDiscontinuityRadius(discontinuityRad)  # Default 4, confidence parameter for gaps/holes in disparity
    wls.setSigmaColor(sigma)

    # roi = cv.getValidDisparityROI(roiL, roiR, minDisp, nDisp, bSize)
    
    ### Compute raw disparity from both sides
    dispL = stereoL.compute(grayL, grayR)
    dispR = stereoR.compute(grayR, grayL)

    ### Filter raw disparity using weighted least squares based smoothing
    dispFinal = wls.filter(dispL, imgL, None, dispR)
    dispFinal = ximgproc.getDisparityVis(dispFinal)    


    paramsVals = [sigma, lam, 
            stereoL.getNumDisparities(), stereoL.getBlockSize(), 
            stereoL.getPreFilterCap(), stereoL.getSpeckleRange()]

    ### Show results
    # display_disparity(imgL, dispL, dispFinal, imgId, paramsVals)

    ''' Map disparity values to depth as 3D point cloud '''
    points3d = cv.reprojectImageTo3D(dispFinal, Q, ddepth=cv.CV_32F, handleMissingValues=True)

    ''' Filter obstacles, compute occupancy grid, find path '''
    find_path(imgId, nDisp, points3d, dispFinal, paramsVals)


def find_path(imgId, nDisp, points3d, disparityMap, params):
    xx, yy, zz = points3d[:,:,0], points3d[:,:,1], points3d[:,:,2]
    xx, yy, zz = np.clip(xx, -18, 38), np.clip(yy, -25, 25), np.clip(zz, 0, 120)

    ''' Filter obstacles above ground/floor plane '''
    obs = zz[200:y_floor,:]
    #xx = xx[:,nDisp:]
    #yy = xx[:,nDisp:]

    ''' Construct occupancy grid '''
    obstacles = np.amin(obs, 0, keepdims=False)
    x, y = np.meshgrid(np.arange(0, obstacles.shape[0]), np.arange(0, np.max(obstacles)))

    # print(obstacles)
    occupancy_grid = np.where(y >= obstacles, 0, 1)
    occupancy_grid[:, :nDisp] = 0

    far_zy, far_zx = np.unravel_index(np.argmax(np.flip(occupancy_grid)), occupancy_grid.shape)
    far_zx = (zz.shape[1]-1) - far_zx
    far_zy = occupancy_grid.shape[0] - far_zy - 1
    near_zy = occupancy_grid.shape[0] - 1
    # print(f'Far_Z (x,y): {far_zx},{far_zy}')


    ''' Compute corresponding X,Y pixel positions of 3D Depth points '''
    xloc = np.where((xx <= 0.05) & (xx >= -0.05))
    yloc = np.where((yy <= 3.1) & (yy >= 2.9))
    xcenter = np.mean(xloc[1], dtype=np.int32)
    ycenter = np.mean(yloc[0], dtype=np.int32)
    # ycenter = 400
    zoffset = np.amin(obstacles)
    print('{}: xcenter={}    ycenter={}    zforward={}'.format(imgId, xcenter, ycenter, zoffset))


    ''' A* path-finding config and computation '''
    grid = Grid(matrix=occupancy_grid)
    start = grid.node(xcenter, 1)
    end = grid.node(far_zx, far_zy)

    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    path, runs = finder.find_path(start, end, grid)
    #print(path)
    # zstart = zz[479, xcenter]
    coords = np.array([(x, 5, z) for x, z in path], dtype=np.int32)
    # plt.plot(coords[:,0], coords[:,2], 'r')


    ''' Map X,Y pixel positions to world-frame for cv.projectPoints() '''
    zworld = (coords[:,2]+zoffset)      # + 9
    ymax = yy.shape[0] - 1
    yrange = np.linspace(ymax, y_floor, num=len(coords), dtype=np.int32)

    xworld = xx[yrange, coords[:,0]]
    yworld = yy[yrange, coords[:,0]]

    cf = np.array([xworld, yworld, zworld]).T


    ''' Reproject 3D world-frame points back to unrectified 2D points'''
    rvec, jacobian = cv.Rodrigues(R)
    pr, jacobian2 = cv.projectPoints(np.float32(cf), rvec, T, CL, DL)
    pr = np.squeeze(pr, 1)
    py = pr[:,1]
    px = pr[:,0]+nDisp


    ''' Update figure (final results) '''
    imL = cv.imread(join(r'L_stream', imgId))
    imL = cv.cvtColor(imL, cv.COLOR_BGR2RGB)

    plt.clf()
    params[-2] = ycenter
    params[-1] = str(zoffset)[0:5]
    paramsText = 'Lambda={}; Sigma={}; nDisp={}; bSize={}; yCenter={}; zForward={}'.format(*params)

    plt.suptitle(imgId)
    plt.gcf().text(x=0.1, y=0.05, s=paramsText, fontsize='small')
    plt.subplot(221); plt.imshow(imL); plt.title('Planned Path (Left Camera)')
    plt.scatter(px, py, np.flip(cf[:,2]), c=cf[:,2], cmap=plt.cm.plasma_r)
    plt.xlim([0, 640]); plt.ylim([480, 0])

    ax = plt.gcf().add_subplot(222, projection='3d')
    ax.azim = 90; ax.elev = 160; ax.set_box_aspect((8, 6, 6))
    ax.plot_surface(xx,yy,zz, cmap=plt.cm.viridis_r, linewidth=0, antialiased=False)
    ax.set_xlabel('Azimuth (X)'); ax.set_ylabel('Elevation (Y)'); ax.set_zlabel('Depth (Z)')
    ax.invert_xaxis(); ax.invert_zaxis(); ax.set_title('Planned Path (wrt. world-frame)')
    ax.scatter3D(cf[:,0],cf[:,1],cf[:,2], c=cf[:,2], cmap=plt.cm.plasma_r)


    plt.subplot(223); plt.imshow(disparityMap); plt.title('WLS Filtered Disparity Map'); 
    plt.subplot(224); plt.imshow(occupancy_grid); plt.ylim([0, 120]); plt.xlim([0, occupancy_grid.shape[1]])
    plt.title('2D Obstacle View (Z-depth vs. X)')
    plt.plot(coords[:,0], coords[:,2], 'r')
    plt.title('Steps={},  Path Length={}'.format(runs, len(path)))


def display_disparity(origImg, dispRaw, dispWLS, imgName, paramsVals):
    ''' Helper function to show figure with some parameters '''

    # Show parameters on figure for debugging
    paramsText = 'Lambda={}\nSigma={}\nnDisp={}\nbSize={}\npfCap={}\nsRange={}'.format(*paramsVals)

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.3)
    plt.suptitle(imgName)
    plt.gcf().text(x=0.15, y=0.8, s=paramsText)

    origImg = cv.cvtColor(origImg, cv.COLOR_BGR2RGB)

    plt.subplot(121); plt.imshow(origImg)
    #plt.subplot(121); plt.imshow(dispRaw, 'gray'); plt.title('Raw Disparity')
    plt.subplot(122); plt.imshow(dispWLS); plt.title('WLS Filter')
    plt.tight_layout()

main()