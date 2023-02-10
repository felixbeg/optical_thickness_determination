# this program lets you select rectangles and returns height averages of the selected area.
# it is designed to work with the txt files from the processed gwyddion image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets  
from tkinter import filedialog as fd
from datetime import datetime
import sys
import pathlib

class AFM:
    def __init__(self):
        self.directory_name = fd.askdirectory()
        self.filename = self.directory_name.split('/')[-1]+' Z C text'
        
        # opwning afm file for parameters of scan
        self.file = open(f'{self.directory_name}/{self.filename}', 'r')
        text = [line for line in self.file]
        xwidth = float(text[1].split(' ')[2])
        ywidth = float(text[2].split(' ')[2])
        xunit = text[1].split(' ')[3]
        yunit = text[2].split(' ')[3]
        xunit = xunit.strip('Â')
        yunit = yunit.strip('Â')
        # zunit = text[3].split(' ')[3]
        zunit = 'nm' # z data is multiplied with 10^9
        self.file.close()

        file = open(f'{self.directory_name}/{self.filename}', 'r')
        self.image = np.loadtxt(file, skiprows=4, dtype=float)*1e9 # z data in nm
        file.close()

        # mainly for formatting image in right way
        info_file = self.directory_name.split('/')[-1]
        file = open(f'{self.directory_name}/{info_file}.txt', 'r')
        info = []
        for line in file:
            info.append(line)
        xarea = float(info[7].split()[7]) # in µm
        yarea = float(info[7].split()[8]) # in µm
        xpixel = float(info[8].split()[7]) 
        ypixel = float(info[8].split()[8]) 
        file.close()
        
        x_pixel_ratio = (xpixel/xarea)/(ypixel/yarea)
        if x_pixel_ratio < 1:
            y_pixel_ratio = 1/x_pixel_ratio
            self.image = np.repeat(self.image, int(y_pixel_ratio), axis=1)
        else:
            self.image = np.repeat(self.image, int(x_pixel_ratio), axis=0) # fixed by option extent in imshow()
        
        # for setting the axes correctly
        xdim = len(self.image[0, :])
        ydim = len(self.image[:, 0])
        xticks = np.arange(int(xwidth)+1)
        str_xticks = [str(tick) for tick in xticks]
        yticks = np.arange(int(ywidth)+1)
        str_yticks = [str(tick) for tick in yticks]

        # initiate the plot
        fig, ax = plt.subplots()

        ax.set_title('AFM scan hBN - height measurement.')
        ax.set_xticks(np.array(xdim*xticks/xwidth, dtype=int), str_xticks)
        ax.set_yticks(np.array(ydim*yticks/ywidth, dtype=int), str_yticks)
        ax.set_xlabel(f'$x$/{xunit}')
        ax.set_ylabel(f'$y$/{yunit}')
        image = ax.imshow(self.image, cmap='gray')
        fig.colorbar(image, label=f'$z$/{zunit}')

        # global boolean for changing text files and variables for calculation of thickness
        self.hbn = True
        self.hbn_height = 0
        self.std_hbn_height = 0
        self.sub_height = 0
        self.std_sub_height = 0


        # change between hBN and substrate rectangle
        def toggle_selector(event):
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
        

        print('\nRectangles for substrate and hBN are currently overlapping. To move only one rectangle, press the \'w\'-key.\n')


        # this function is needed to interact with the plot, e.g. drawing the rectangles
        def onselect(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            selected_area = self.image[int(y1):int(y2), int(x1):int(x2)]

            mu_height = np.mean(selected_area)
            sigma_height = np.std(selected_area)

            if self.hbn:
                self.hbn_height = mu_height
                self.std_hbn_height = sigma_height
            else:   
                self.sub_height = mu_height
                self.std_sub_height = sigma_height

            # calculate thickness
            self.thickness = self.hbn_height - self.sub_height
            self.std_thickness = np.sqrt(self.std_hbn_height**2+self.std_sub_height**2)
            
            # print result
            print(f'\nHeight of hBN: ({self.hbn_height:.2f}+-{self.std_hbn_height:.2f}) {zunit}')
            print(f'Height of substrate: ({self.sub_height:.2f}+-{self.std_sub_height:.2f}) {zunit}')
            print(f'Thickness of hBN-flake: ({self.thickness:.2f}+-{self.std_thickness:.2f}) {zunit}\n')

        # connect callback functions
        fig.canvas.mpl_connect('key_press_event', toggle_selector)
        props_hbn = dict(facecolor='blue', alpha=0.5)
        props_sub = dict(facecolor='green', alpha=0.5)
        rect_hbn = mwidgets.RectangleSelector(ax, onselect, interactive=True, props=props_hbn)
        rect_sub = mwidgets.RectangleSelector(ax, onselect, interactive=True, props=props_sub)
        rect_hbn.set_active(True)
        rect_sub.set_active(False)

        plt.show()

        print('Do you want to save the plot and/or the measured values?')
        print('Type \'y\' to save')
        print('Type \'n\' to discard\n')

        timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')+'_A'
        folder = './measurements'
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        
        # interact with program in terminal
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
                file.write(f'thickness/{zunit} sd_thickness/{zunit} height_hbn/{zunit} sd_height_hbn/{zunit} height_sub/{zunit} sd_height_sub/{zunit}\n')
                file.write(f'{self.thickness} {self.std_thickness} {self.hbn_height} {self.std_hbn_height} {self.sub_height} {self.std_sub_height}\n')
                file.close()
                print('\nFiles saved.\n')
                break
            if 'n' == entry1.rstrip():
                print('\nFiles discarded.\n')
                break
            else:
                print('\n Type \'y\' or \'n\'!\n')


while 1:
    run = AFM()
