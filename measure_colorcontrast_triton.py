from arena_api.system import system
from arena_api.buffer import *

import ctypes
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from matplotlib_scalebar.scalebar import ScaleBar
import cv2
import argparse
import sys
from datetime import datetime
import pathlib
from tkinter import filedialog as fd

# parameters for image correction can be added in the terminal as postional argument
parser = argparse.ArgumentParser(description='Stream camera, take snapshot and evaluate color of hBN and substrate.')
parser.add_argument('-a', '--gain', nargs=1, type=float, metavar='alpha', help='Gain in dB') # this is not in dB yet
parser.add_argument('-b', '--bias', nargs=1, type=float, metavar='beta', help='Bias')
parser.add_argument('-g', '--gamma', nargs=1, type=float, metavar='gamma', help='Gamma Correction')
args = parser.parse_args()

# parameters for immage correction
if args.gain:
    alpha = args.gain[0]
else:
    alpha = 1
if args.bias:
    beta = args.bias[0]
else:
    beta = 0
if args.gamma:
    gamma = args.gamma[0]
else:
    gamma = 1
        

# function for gamma-correction
# now with lut since it's supposed to be faster 
table_gamma = np.array([(i/255)**(1/gamma)*255 for i in range(0, 256)]).astype('uint8')
def gamma_correct(array):
    return cv2.LUT(array, table_gamma)


# defining functions for contrast adjustments
# function for gain alpha and bias beta
table_alpha_beta = np.array([alpha*i+beta for i in range(0, 256)]).astype('uint8')
table_alpha_beta = np.where(table_alpha_beta<255, table_alpha_beta, 255)
def adjust_contrast(array):
    return cv2.LUT(array, table_alpha_beta)


# function for autobalancing white, only for streaming
def balance_white(array):
    R_avg = np.average(array[:, :, 0])
    G_avg = np.average(array[:, :, 1])
    B_avg = np.average(array[:, :, 2])

    CorrR = G_avg/R_avg
    CorrB = G_avg/B_avg

    corr_img = np.zeros(np.shape(array))
    corr_img[:, :, 1] = array[:, :, 1]
    corr_img[:, :, 0] = CorrR*array[:, :, 0]
    corr_img[:, :, 2] = CorrB*array[:, :, 2]

    corr_img = corr_img/np.amax(corr_img)
    return corr_img


# some calculations and definitions for scalebar in cv2 and matplot
dx_obj5 = 1.3024119800598069 # µm
dx_obj10 = 0.6513586780443329 # µm
dx_obj20 = 0.3232413688296831 # µm
dx_obj50 = 0.129080422000821 # µm
dx_obj100 = 0.06401050835553976 # µm
pixel_widths = np.array([dx_obj100, dx_obj50, dx_obj20, dx_obj10, dx_obj5])
objective = [100, 50, 20, 10, 5]
scales = np.array([20, 30, 30, 100, 400]) 
lengths = scales/pixel_widths
linewidth = 2
start_point = 20
end_points = lengths+start_point-linewidth/2


class LucidCamera:
    def __init__(self, stream=1):
        if args.gain:
            self.alpha = args.gain[0]
        if args.bias:
            self.beta = args.bias[0]
        if args.gamma:
            self.gamma = args.gamma[0]
        self.stream = stream
        self.path_results_folder = './measurements'


    # create device, waits 20 seconds for device to be connected
    def _CreateDeviceWithTries(self):
        tries = 0
        tries_max = 2
        sleep_time_secs = 10
        devices = None
        while tries < tries_max:
            devices = system.create_device()
            if not devices:
                print(
                    f'Try {tries+1} of {tries_max}: '
                    f'wait {sleep_time_secs} sceonds for connection!'
                )
                for sec_count in range(sleep_time_secs):
                    time.sleep(1)
                    print(
                        f'{sec_count+1} seconds passed ',
                        '.'*sec_count, end='\r'
                    )
                tries += 1
            else:
                return devices
        else:
            raise Exception(f'No device found! Please connect device and run again.')


    def Create(self):
        print('\nProgram started\n')

        devices = self._CreateDeviceWithTries()
        self.device = devices[0]


    def Setup(self):
        print('Setting up device.\n')
        self.nodemap = self.device.nodemap
        self.nodes = self.nodemap.get_node(['Width', 'Height', 'PixelFormat', 'AcquisitionFrameRateEnable', 'ExposureAuto', 'ExposureTime', 'GainAuto', 'GammaEnable', 'BalanceWhiteEnable', 'BalanceWhiteAuto'])
        self.initial_nodes = self.nodes

        # formatting
        self.nodes['Width'].value = 1936
        self.nodes['Height'].value = 1464
        self.nodes['PixelFormat'].value = 'RGB8'


        # control nodes
        self.nodes['AcquisitionFrameRateEnable'].value = False
        self.nodes['ExposureAuto'].value = 'Off'
        self.nodes['GainAuto'].value = 'Off'
        self.nodes['GammaEnable'].value = False
        self.nodes['BalanceWhiteEnable'].value = False
        self.nodes['BalanceWhiteAuto'].value = 'Off'

        # Stream nodemap
        tl_stream_nodemap = self.device.tl_stream_nodemap
        tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
        tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
        tl_stream_nodemap['StreamPacketResendEnable'].value = True


    def Destroy(self):
        self.nodes = self.initial_nodes
        system.destroy_device(self.device)


    def Stream(self):
        num_channels = 3 # depends on pixel format, for RGB -> 3

        # control panel of exposure time
        expo_min = 1500
        expo_max = 10000


        def _set_exposure_time(val: float):
            self.exposure_time = float(val)
            if self.exposure_time > self.nodes['ExposureTime'].max:
                self.nodes['ExposureTime'].value = self.nodes['ExposureTime'].max
            elif self.exposure_time < self.nodes['ExposureTime'].min:
                self.nodes['ExposureTime'].value = self.nodes['ExposureTime'].min
            else:
                self.nodes['ExposureTime'].value = self.exposure_time
            #print(self.nodes['ExposureTime'].value)

        
        cv2.namedWindow('Exposure Time t/microsecond')
        cv2.resizeWindow('Exposure Time t/microsecond', 1000, 50)
        cv2.createTrackbar('t', 'Exposure Time t/microsecond', expo_min, expo_max, _set_exposure_time)
        self.nodes['ExposureTime'].value = 1500.0

        prev_frame_time = 0
        curr_frame_time = 0
        self.mag = 1
        print('\nStream started.\n')
        print('Choose magnification with keys: \'1\', \'2\', \'3\', \'4\' or \'5\'.\n')
        with self.device.start_stream():
            while True:
                curr_frame_time = time.time() # Used to display FPS on stream
                
                # get image from camera
                buffer = self.device.get_buffer()
                item = BufferFactory.copy(buffer)
                self.device.requeue_buffer(buffer)
                buffer_bytes_per_pixel = int(len(item.data)/(item.width * item.height))
                array = (ctypes.c_ubyte * num_channels * item.width * item.height).from_address(ctypes.addressof(item.pbytes))
                npndarray = np.ndarray(buffer=array, dtype=np.uint8, shape=(item.height, item.width, buffer_bytes_per_pixel))
                self.image = np.copy(npndarray)

                # image correction
                if args.gamma:
                    npndarray = gamma_correct(npndarray)
                if args.gain or args.bias:
                    npndarray = adjust_contrast(npndarray)

                # text to put on broadcasted image
                max_R = np.amax(npndarray[:, :, 0])
                max_G = np.amax(npndarray[:, :, 1])
                max_B = np.amax(npndarray[:, :, 2])
                npndarray = cv2.cvtColor(npndarray, cv2.COLOR_RGB2BGR) # convert for image to be shown correctly in cv2
                fps = f'FPS: {1/(curr_frame_time-prev_frame_time):.1f}'
                max = f'Max. (R, G, B): ({max_R:.0f}, {max_G:.0f}, {max_B:.0f})'
                string = f'{scales[self.mag-1]} um ({10*objective[self.mag-1]}x magnification)'
                cv2.rectangle(npndarray, (start_point, 20), (int(end_points[self.mag-1]), 20), (100, 255, 0), linewidth)
                cv2.rectangle(npndarray, (start_point, 15), (start_point, 25), (100, 255, 0), linewidth)
                cv2.rectangle(npndarray, (int(end_points[self.mag-1]), 15), (int(end_points[self.mag-1]), 25), (100, 255, 0), linewidth)
                cv2.putText(npndarray, string, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(npndarray, max, (10, 660), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(npndarray, fps, (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

                # show image
                cv2.imshow('Camera currently streaming, press \'s\'-key to stop stream/take snapshot.', npndarray)

                # clean up
                BufferFactory.destroy(item)
                prev_frame_time = curr_frame_time

                # Break if s-key (snapshot) is pressed
                key = cv2.waitKey(1)
                if key == ord('1'):
                    self.mag = 1
                    npndarray = np.copy(self.image)
                if key == ord('2'):
                    self.mag = 2
                    npndarray = np.copy(self.image)
                if key == ord('3'):
                    self.mag = 3
                    npndarray = np.copy(self.image)
                if key == ord('4'):
                    self.mag = 4
                    npndarray = np.copy(self.image)
                if key == ord('5'):
                    self.mag = 5
                    npndarray = np.copy(self.image)
                if key == 115:
                    break

            self.device.stop_stream()
            cv2.destroyAllWindows()
        print('\nStream stopped.\n') 


    # reads numpy binaray file and load it as array/image
    def Read(self):
        filename = fd.askopenfilename()
        self.image = np.load(filename)
        filename_txt = filename.strip('npy')+'txt'
        self.exposure_time = np.loadtxt(filename_txt, skiprows=1, max_rows=1)

    def Snap(self):
        # get last image of stream
        print("\nTaking snapshot.\n")

        # setup plot
        print('To break streaming-/snapping-loop, press \'b\'-key.')
        fig, ax = plt.subplots()
        if self.stream:
            scalebar = ScaleBar(pixel_widths[self.mag-1], 'um', frameon=1) 
        else:
            scalebar = ScaleBar(pixel_widths[0], 'um', frameon=1) 
        ax.add_artist(scalebar)
        ax.set_title('Select rectangles for color contrast.\n Blue = hBN, green = substrate.')
        ax.imshow(self.image)


        # function for breaking streaming/snapping loop
        def _switch_off(event):
            global boolean
            if event.key == 'b':
                if boolean:
                    boolean = False
                    print('Camera will stop streaming after saving/discarding files.\n')
                else:
                    boolean = True 
                    print('Camera will continue streaming after saving/discarding files.\n')


        # change between hBN and substrate rectangle
        def _toggle_selector(event):
            if event.key == 'w':
                name = type(rect_hbn).__name__
                if rect_hbn.active:
                    print(f'\n{name} for hBN deactivated.')
                    print(f'{name} substrate activated.\n')
                    
                    rect_hbn.set_active(False)
                    rect_sub.set_active(True)
                    self.hbn = False
                else:
                    print(f'\n{name} for hBN activated.')
                    print(f'{name} substrate deactivated.\n')
                    
                    rect_hbn.set_active(True)
                    rect_sub.set_active(False)
                    self.hbn = True
            

        # callback function for rectangles
        self.hbn = True
        self.sub_R = 1 # to prevent attribute error on first click, initialize with arbitrary numbers
        self.sd_sub_R = 1
        self.sub_G = 1
        self.sd_sub_G = 1
        self.sub_B = 1
        self.sd_sub_B = 1
        self.sub_colors = 'No values yet.'

        def _onselect(eclick, erelease):  
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata

            selected_area = self.image[int(y1):int(y2), int(x1):int(x2)]
            
            # calculate mean color value of each channel
            mean_red = np.mean(selected_area[:, :, 0])
            mean_green = np.mean(selected_area[:, :, 1])
            mean_blue = np.mean(selected_area[:, :, 2])
            std_red = np.std(selected_area[:, :, 0])
            std_green = np.std(selected_area[:, :, 1])
            std_blue = np.std(selected_area[:, :, 2])
            
            if self.hbn:
                self.hbn_R = mean_red
                self.sd_hbn_R = std_red
                self.hbn_G = mean_green
                self.sd_hbn_G = std_green
                self.hbn_B = mean_blue
                self.sd_hbn_B = std_blue 
                self.hbn_colors = f'({mean_red:.2f}+-{std_red:.2f}, {mean_green:.2f}+-{std_green:.2f}, {mean_blue:.2f}+-{std_blue:.2f})'
            else:
                self.sub_R = mean_red
                self.sd_sub_R = std_red
                self.sub_G = mean_green
                self.sd_sub_G = std_green
                self.sub_B = mean_blue
                self.sd_sub_B = std_blue
                self.sub_colors = f'({mean_red:.2f}+-{std_red:.2f}, {mean_green:.2f}+-{std_green:.2f}, {mean_blue:.2f}+-{std_blue:.2f})'

            # calculate color contrast
            self.contrast_R = (self.hbn_R-self.sub_R)/self.sub_R
            self.sd_contrast_R = np.sqrt((self.sd_hbn_R/self.sub_R)**2+(self.sd_sub_R*self.hbn_R/self.sub_R**2)**2)
            self.contrast_G = (self.hbn_G-self.sub_G)/self.sub_G
            self.sd_contrast_G = np.sqrt((self.sd_hbn_G/self.sub_G)**2+(self.sd_sub_G*self.hbn_G/self.sub_G**2)**2)
            self.contrast_B = (self.hbn_B-self.sub_B)/self.sub_B
            self.sd_contrast_B = np.sqrt((self.sd_hbn_B/self.sub_B)**2+(self.sd_sub_B*self.hbn_B/self.sub_B**2)**2)

            # print results
            print('hBN RGB: '+self.hbn_colors)
            print('Substrate RGB: '+self.sub_colors+'\n')
            print(f'Contrast_R : {self.contrast_R:.2f}+-{self.sd_contrast_R:.2f}')
            print(f'Contrast_G : {self.contrast_G:.2f}+-{self.sd_contrast_G:.2f}')
            print(f'Contrast_B : {self.contrast_B:.2f}+-{self.sd_contrast_B:.2f}\n')

        # connect and setu up callback functions
        fig.canvas.mpl_connect('key_press_event', _toggle_selector)
        fig.canvas.mpl_connect('key_press_event', _switch_off)
        props_hbn = dict(facecolor='blue', alpha=0.5)
        props_sub = dict(facecolor='green', alpha=0.5)
        rect_hbn = mwidgets.RectangleSelector(ax, _onselect, interactive=True, props=props_hbn)
        rect_sub = mwidgets.RectangleSelector(ax, _onselect, interactive=True, props=props_sub)
        rect_hbn.set_active(True)
        rect_sub.set_active(False)

        print('\nRectangle for hBN currently activated. To change between rectangles, press the \'r\'-key.')
        print('Note that color contrast will only be sensible with second rectangle selected.\n')

        plt.show()

        # after finalising evaluation, save/discard data
        print('Do you want to save the contrast values and plot?')
        print('Type \'y\' to save')
        print('Type \'n\' to discard\n')

        timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')+'_N'
        folder = self.path_results_folder
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        
        # part for interacting with the program in the terminal
        for entry1 in sys.stdin:
            if 'y' == entry1.rstrip():
                print('\nDo you want extra description?')
                print('Type your description (e.g. sample001_flake001) if yes, enter \'n\' if no\n')
                for entry2 in sys.stdin:
                    if 'n' == entry2.rstrip():
                        file = open(folder+'/'+timestamp+'.txt', 'w')
                        fig.savefig(folder+'/'+timestamp+'.png')
                        np.save(folder+'/'+timestamp, self.image)
                    else:
                        file = open(folder+'/'+timestamp+'_'+entry2.rstrip()+'.txt', 'w')
                        fig.savefig(folder+'/'+timestamp+'_'+entry2.rstrip()+'.png')
                        np.save(folder+'/'+timestamp+'_'+entry2.rstrip(), self.image)
                    break
                if stream:
                    exposure_time_2_string = self.nodes['ExposureTime'].value
                else:
                    exposure_time_2_string = self.exposure_time
                file.write(f'Exposure Time\n{exposure_time_2_string}\n')
                file.write(f'Channel contrast sd_contrast mean_hbn std_hbn mean_sub std_hbn\n')
                file.write(f'R {self.contrast_R} {self.sd_contrast_R} {self.hbn_R} {self.sd_hbn_R} {self.sub_R} {self.sd_sub_R}\n')
                file.write(f'G {self.contrast_G} {self.sd_contrast_G} {self.hbn_G} {self.sd_hbn_G} {self.sub_G} {self.sd_sub_G}\n')
                file.write(f'B {self.contrast_B} {self.sd_contrast_B} {self.hbn_B} {self.sd_hbn_B} {self.sub_B} {self.sd_sub_B}\n')
                file.close()
                print('\nFiles saved.\n')
                break
            if 'n' == entry1.rstrip():
                print('\nFiles discarded.\n')
                break
            else:
                print('\n Type \'y\' or \'n\'!\n')
        
        print('\nProgram finished.\n')


# choose what part of the program you want to use here
stream = 0 # True for streaming, False for opening old image
Camera = LucidCamera(stream=stream)

if stream:
    boolean = True # will be set to False when 'b' is pressed in Snap()-view
    Camera.Create()
    Camera.Setup()
    while boolean:
        print('To break streaming/snapping loop, press \'b\'-key in Snap()-view.')
        Camera.Stream()
        Camera.Snap()
    Camera.Destroy()
else:
    Camera.Read()
    Camera.Snap()
