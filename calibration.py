import numpy as np
import cv2 as cv
from os import listdir
from os.path import isfile, join, splitext
import matplotlib.pyplot as plt

def sort_id(e):
  return int(e.split('.')[0])


ROOT = r'C:\Users\BW\Documents\Python Scripts\Senior Design'
output_id = '7.jpg'
file_list = [i for i in listdir(join(ROOT, 'L'))]
file_list.sort(key=sort_id)

### termination criteria
criteria_calib = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1000, 1e-6)
flags_thresh = (cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
flags_stereo_calib = (cv.CALIB_FIX_INTRINSIC)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
print(objp.shape)
# Store computed points
objPts = []
imgPL, imgPR = [], []

imgSize = (640, 480) # (768, 1024), 1296x972


for f in file_list:
    fname = str(f)

    imgL = cv.imread(join(ROOT, 'L', fname))
    imgR = cv.imread(join(ROOT, 'R', fname))
    grayL = cv.cvtColor(imgL, cv.COLOR_RGB2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_RGB2GRAY)
    # h,w = grayL.shape
    retL, cornersL = cv.findChessboardCorners(grayL, (9,6), flags=flags_thresh)
    retR, cornersR = cv.findChessboardCorners(grayR, (9,6), flags=flags_thresh)

    if retL and retR:
        objPts.append(objp)
        corners2L = cv.cornerSubPix(grayL, cornersL, (5,5), (-1,-1), criteria_calib)
        imgPL.append(corners2L)
        # objPR.append(objp)
        corners2R = cv.cornerSubPix(grayR, cornersR, (5,5), (-1,-1), criteria_calib)
        imgPR.append(corners2R)

        # if f == output_id:
        #     ## Draw and display the corners
        #     cv.imwrite(r'L_' + fname, imgL)
        #     cv.drawChessboardCorners(imgL, (9,6), corners2L, retL)
        #     cv.imwrite(r'GRID_L_' + fname, imgL)

        #     cv.imwrite(r'R_' + fname, imgR)
        #     cv.drawChessboardCorners(imgR, (9,6), corners2R, retR)
        #     cv.imwrite(r'GRID_R_' + fname, imgR)
        #     cv.waitKey(500)
        #     print(f'OUTPUT: {f}')
        #     exit()
    else:
        print(f'{fname}: Corners not found -- (L,R) ({retL},{retR})')
        break



objPts = np.asarray(objPts, np.float32)
imgPL = np.asarray(imgPL, np.float32)
imgPR = np.asarray(imgPR, np.float32)


ret1, C1, D1, R1, T1 = cv.calibrateCamera(objPts, imgPL, imgSize, None, None, criteria=criteria_calib)
ret2, C2, D2, R2, T2 = cv.calibrateCamera(objPts, imgPR, imgSize, None, None, criteria=criteria_calib)

retval,CL,DL,CR,DR,R,T,E,F, = cv.stereoCalibrate(objPts, imgPL, imgPR, 
                                    C1, D1, C2, D2, imgSize, flags=flags_stereo_calib, criteria=criteria_calib)

retval,CLl,DLl,CR,DR,rr,tt,E,F,pve = cv.stereoCalibrateExtended(objPts, imgPL, imgPR, 
                                    C1, D1, C2, D2, imgSize, R, T, flags=flags_stereo_calib, criteria=criteria_calib)                                

# Rectification Transforms, Projection Matrices, ROIs after rectification
RL,RR,PL,PR,Q,validROIL,validROIR = cv.stereoRectify(CL, DL, CR, DR, imgSize, R, T, alpha=0.5, newImageSize=imgSize)


''' Print critical parameters and per-view reprojection error '''
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
print('CameraMatrix_L = \n{}\n\nCameraMatrix_R = \n{}\n'.format(CL, CR))
print('RotationStereo = \n{}\n\nTranslationStereo = \n{}\n'.format(R, T))
print('DistCoeffStereo_L = \n{}\n\nDistCoeffStereo_R = \n{}\n'.format(DL, DR))
print('Q = \n{}\n'.format(Q))

labels = []
for i,e in enumerate(pve):
    print(f'Pair {file_list[i]}: {e}')
    n = file_list[i].split('.')[0]
    labels.append(n)
print(f'\nMSE: {retval:0.4f} ({len(file_list)} image pairs)\n')


''' Display plot of per-view reprojection error '''
fig, ax = plt.subplots()
x_label_pos = np.arange(len(pve))
bar_width = 0.35
rects1 = ax.bar(x_label_pos - bar_width/2, pve[:,0], bar_width, label='Camera 1')
rects2 = ax.bar(x_label_pos + bar_width/2, pve[:,1], bar_width, label='Camera 2')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Mean Squared Error (pixels)')
ax.set_xlabel('Image Pair ID')
ax.set_title('Pixel Reprojection Error per Image Pair')
ax.set_xticks(x_label_pos)
ax.axhline(retval, color="black", linestyle="--", label='Overall MSE='+str(retval)[:4])
ax.set_xticklabels(labels)
ax.legend()

### Refine Rotational and Translational
# rr, j = cv.Rodrigues(R)
# pnp1, rv1, tv1 = cv.solvePnP(objPts[0], imgPL[0], C1, D1, rvec=rr, tvec=T, useExtrinsicGuess=True, flags=(cv.SOLVEPNP_ITERATIVE))
# print('rv1 {}\ntv1 {}'.format(rv1, tv1))

# pnp2, rv2, tv2, inliers = cv.solvePnPRansac(objPts[0], imgPL[0], C1, D1, rvec=rr, tvec=T, useExtrinsicGuess=True)
# print('rv2 {}\ntv2 {}'.format(rv2, tv2))

# rv3, tv3 = cv.solvePnPRefineLM(objPts[0], imgPL[0], C1, D1, rr, T)
# print('rv3 {}\ntv3 {}'.format(rv3, tv3))


''' Rectification mapping '''
undistL, rectifL = cv.initUndistortRectifyMap(CL, DL, RL, PL, imgSize, cv.CV_32FC1)
undistR, rectifR = cv.initUndistortRectifyMap(CR, DR, RR, PR, imgSize, cv.CV_32FC1)


''' Preview rectification & remap '''
preview_img = '7.jpg'
img1 = cv.imread(join(ROOT, 'L', preview_img))
img2 = cv.imread(join(ROOT, 'R', preview_img))
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

img11 = cv.remap(img1, undistL, rectifL, cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT)
img22 = cv.remap(img2, undistR, rectifR, cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT)

plt.subplot(221); plt.imshow(img11)
plt.subplot(222); plt.imshow(img22)

x, y, w, h = validROIL
img11 = img11[y:y+h, x:x+w]

x, y, w, h = validROIR
img22 = img22[y:y+h, x:x+w]

dsize = (img11.shape[1], img11.shape[0])
img22 = cv.resize(img22, dsize, interpolation=cv.INTER_CUBIC)

### Show rectified and remapped
plt.subplot(223); plt.imshow(img11); plt.title(img11.shape)
plt.subplot(224); plt.imshow(img22); plt.title(img22.shape)
plt.tight_layout()
plt.show()


''' Write parameters to .txt files '''
prompt = input('Save arrays to file? (y/n): ')

if (prompt == 'y'):
    # np.savetxt(r'Calibration_Files\C1.txt', C1, fmt='%.5e')   # identical to CL
    # np.savetxt(r'Calibration_Files\D1.txt', D1, fmt='%.5e')   # identical to DL
    np.savetxt(r'Calibration_Files\Q.txt', Q, fmt='%.5e')
    np.savetxt(r'Calibration_Files\CmL.txt', CL, fmt='%.5e')
    np.savetxt(r'Calibration_Files\CmR.txt', CR, fmt='%.5e')
    np.savetxt(r'Calibration_Files\DcL.txt', DL, fmt='%.5e')
    np.savetxt(r'Calibration_Files\DcR.txt', DR, fmt='%.5e')
    np.savetxt(r'Calibration_Files\Rtn.txt', R, fmt='%.5e')
    np.savetxt(r'Calibration_Files\ProjL.txt', PL, fmt='%.5e')
    np.savetxt(r'Calibration_Files\ProjR.txt', PR, fmt='%.5e')
    np.savetxt(r'Calibration_Files\Trnsl.txt', T, fmt='%.5e')
    np.savetxt(r'Calibration_Files\umapL.txt', undistL, fmt='%.5e')
    np.savetxt(r'Calibration_Files\rmapL.txt', rectifL, fmt='%.5e')
    np.savetxt(r'Calibration_Files\umapR.txt', undistR, fmt='%.5e')
    np.savetxt(r'Calibration_Files\rmapR.txt', rectifR, fmt='%.5e')
    np.savetxt(r'Calibration_Files\ROIL.txt', validROIL, fmt='%.5e')
    np.savetxt(r'Calibration_Files\ROIR.txt', validROIR, fmt='%.5e')

cv.destroyAllWindows()