import os
from os import listdir
from os.path import join, exists, splitext
import time
from picamera import PiCamera
from gpiozero import Button


CAMERA = r'R'
PATH = join(r'/home/pi/STEREO_VISION', CAMERA + '_calib')
btn = Button(24)        # GPIO 24 (pin # 18)
pair_id = 0

if not exists(PATH):
    os.mkdir(PATH)
elif len(listdir(PATH)) > 0:
    fnames = [int(splitext(i)[0]) for i in listdir(PATH)]
    pair_id = max(fnames) + 1


try:
    with PiCamera(resolution=(640,480)) as camera:
        camera.rotation = 180
        time.sleep(2)

        camera.shutter_speed = camera.exposure_speed
        camera.exposure_mode = 'off'
        g = camera.awb_gains
        camera.awb_mode = 'off'
        camera.awb_gains = g

        # Synchronize capture start between L/R devices
        print('Waiting for button...')
        btn.wait_for_press()
        start = time.time()
        print(f'Starting: {start}')

        # fname = join(PATH, str(pair_id) + '_' + CAMERA + '.jpg')
        fname = join(PATH, str(pair_id) + '.jpg')
        camera.capture(fname, 'jpeg')
        finish = time.time()

    print('Captured in {:.3f} sec'.format(finish-start))
except (Exception, KeyboardInterrupt) as e:
    print('\n>> Process Exception ({})'.format(e))