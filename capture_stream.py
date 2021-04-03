import os
from os import listdir
from os.path import join, exists, splitext
import io
import time
from picamera import PiCamera
from gpiozero import Button


CAMERA = r'R'   
PATH = join(r'/home/pi/STEREO_VISION', CAMERA + '_calib')
btn = Button(24)        # GPIO 24 (pin # 18)

if not (exists(PATH)):
    os.mkdir(PATH)


class SplitFrames(object):
    def __init__(self):
        self.frame_num = 0
        self.output = None


    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # Start of new frame; close the old one (if any) and
            # open a new output
            if self.output:
                self.output.close()
                # print(self.frame_num, 'closed')

            self.frame_num += 1

            self.output = io.open(join(PATH, str(self.frame_num) + '.jpg'), 'wb')
            self.output.write(buf)
        else:
            print('Bad buffer {}'.format(self.frame_num))


try:
    with PiCamera(resolution=(640,480), framerate=10) as camera:
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

        output = SplitFrames()
        camera.start_recording(output, format='mjpeg')
        camera.wait_recording(10)
        camera.stop_recording()
        finish = time.time()

    print('Captured {} frames at {:.2f}fps'.format(output.frame_num, output.frame_num / (finish - start)))
except (Exception, KeyboardInterrupt) as e:
    print('\n>> Process Exception ({})'.format(e))