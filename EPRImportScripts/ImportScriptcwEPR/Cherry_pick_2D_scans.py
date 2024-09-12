# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:08:00 2024

@author: Boris.Dzikovski
"""

'Use the commented part if you do not want to use the file dialog'
# "GIVE YOUR FOLDER/FILENAME HERE WITHOUT EXTENSION"
# '*****************************************************************'
# folder='C:\Data\Y2024\Kelly_May_16_2024/'
# filename="Tempol_10mcm_W_23dB_outbreak test"

# '*****************************************************************'


import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


def find_line(info_file, key_word):
    "looks for key words like SweepWidth in the DSC file"
    n=0
    for line in info_file:
            'For each line of the DSC file, checks if the line contains' 
            'the desired word like CenterField or FieldSweep'
            if key_word in line:
                return n
            else:
                n=n+1
  
def extract_value (string):
    'extracts a number from a string in the DSC file'
    'if the key_word goes first and the number goes next '
    split_line=string.split()
    number=float(split_line[1])
    return number 

def remove_line(info_file, key_word):
    'removes a line containing a certain key word from the DSC file'
    'if the line is not found nothing happens'
    try:      
        line_number=find_line(info, key_word)
        info.pop(line_number)
        return info_file
    except:
        pass

def baseline (A):

    '''baseline function
    takes numbs points on each end of the spectrum
    draws a linear baseline between the middles of the numbs-points intervals
    and subtracts it from the spectrum
    '''
    bsl1 = 0
    bsl2 = 0
    NUMPT=len(A)

    numbs = 8 
    for i in range(0,numbs-1):      
        bsl1 = bsl1 + A[i]

    bsl1 = bsl1 / (numbs-1)
    
    
    for i in range(NUMPT-numbs+1,NUMPT):  
        bsl2 = bsl2 + A[i]

    bsl2 = bsl2 / (numbs-1)
    
        
   
    
    for i in range(0,NUMPT):
        A[i] = A[i] - bsl1 + (bsl1 - bsl2) * (i - numbs/2)/(NUMPT - numbs)
        
        
    return(A)

'**********************************************************************************'
'MAIN PROGRAM BEGINS'
'using the file dialog to load the experimental 2D file'
win = tk.Tk()
'Makes the file dialog window topmost (it is useful if you use Spyder)'
win.wm_attributes('-topmost', True)

win.withdraw()

filename = tk.filedialog.askopenfilename( title= "Please select a DSC or DTA file:")

print("'"+ filename +"'")
win.destroy()


file_in_folder=filename[:-4]

'Making *.DTA and *.DSC file names'
 
Data_file=file_in_folder+".DTA"
Parm_file=file_in_folder+".DSC"


'reading the DTA file as binary'
with open(Data_file, 'rb') as DTAread:
    sp_value=np.frombuffer(DTAread.read(), dtype='>d')
    
'reading the DSC file as text'   
with open(Parm_file, 'rt') as DSCread:
    info=DSCread.readlines()

'finding the line numbers containing the center field and sweep values. Also looks for the' 
'the line containing the number of points in each 1D layer in the DSC file' 
line_number_CF=find_line(info,'CenterField')
line_number_Sweep=find_line(info,'SweepWidth') 
line_number_points_in_scan=find_line(info,'XPTS') 

'finding info line in the DSC file containing these values'
c_field_line=info[line_number_CF]
sweep_line=info[line_number_Sweep]
points_in_scan_line=info[line_number_points_in_scan]

'extracting numerical values of center field and sweep'
center_field=extract_value(c_field_line)
sweep=extract_value(sweep_line)
'extracting the number of points in a 1D scan'
points_in_scan=int(extract_value(points_in_scan_line))

'making the field column'
B0_start=center_field-sweep/2
B0_end=center_field+sweep/2

field=np.linspace(B0_start, B0_end, num=points_in_scan)


spectrum_1D=np.zeros(points_in_scan)



all_points=len(sp_value)

'number_of_scans is the second dimension of the spectrum, it is also given in'
'the info DCS file in the YPTS and A2RS lines'
number_of_scans=int(all_points/points_in_scan)

scan_count=0

'PLOTTING each scan and asking if to keep or or not, counting good scans which been kept'
'To keep the scan just press "Enter", to dump the scan input "n" or"N" followed by "Enter"'
for i in range(0 , number_of_scans):
    scan_count=scan_count+1
    fig=plt.figure()
    # plt. plot(field[i*int(points_in_scan):(i+1)*int(points_in_scan)-1], sp_value[i*int(points_in_scan):(i+1)*int(points_in_scan)-1], color='blue')
    plt. plot(field[0:points_in_scan-1], sp_value[i*points_in_scan:(i+1)*points_in_scan-1], color='green')
    plt.show()
    print('Keep the scan number', i+1, '?')
    keep_or_not=input("Type 'N' or 'n' if dumping the scan   ")
 
    
    if keep_or_not!='n' and keep_or_not!='N':
        spectrum_1D=spectrum_1D+sp_value[i*points_in_scan:(i+1)*points_in_scan]
    else:
        scan_count=scan_count-1    
    #print(i+1,scan_count)        
        
    plt.close(fig)

'normalizing the spectrum intensity by dividing by the number of good scans '
spectrum_1D=spectrum_1D/scan_count

'Baseline if needed'
baseline_or_not=True
if baseline_or_not:
    spectrum_1D=baseline(spectrum_1D)

    
plt.figure()
plt.plot(field, spectrum_1D)



'SAVING AS ASCII IF NEEDED'
'Set save_or_not=True to save the DSC file as a two-column ASCII with a ".dat" extension'
save_or_not=True

if save_or_not:
    'saving the file with a *.dat extension in the same folder'

    save_name=file_in_folder+'_1D.dat'

    saved_file = open(save_name, "w")
    'converting lists into numpy arrays'
    field_np=np.array(field)
    sp_np=np.array(spectrum_1D)
    'making a two-column file'
    spectrum=np.column_stack((field_np,sp_np))

    np.savetxt(saved_file, spectrum)


    saved_file.close() 


save_DTA_or_not=True

if save_DTA_or_not:
            
    '**************************************************************'
    'DTA/DSC file is saved under the same name with addition _1D'
    '**************************************************************'

    save_name=file_in_folder+'_1D'
    save_name_DTA=save_name+'.DTA'
    save_name_DSC=save_name+'.DSC'
    bin_file = open(save_name_DTA, 'wb')
    

    'Converting float32 type to float64 and swapping bytes '
    
    
    sp_np=np.array(spectrum_1D) 
    
    sp_np=np.float64(sp_np)
    
    sp_np=sp_np.byteswap()

    'saving as Bruker DTA'
    bin_file.write(sp_np)


    bin_file.close()   
    '**************************************************************'
    'Making a new DSC file without lines containing 2D information'
    '**************************************************************'
    'Finding line numbers in the DSC file which are related to 2D, popping or modifying them'
    line_YTYP_number=find_line(info, 'YTYP')
    info[line_YTYP_number]='YTYP	NODATA\n'
    
    remove_line(info, 'YFMT')
   
    remove_line(info, 'YPTS')
    
    remove_line(info, 'YMIN')
    
    remove_line(info, 'YWID')
    
    remove_line(info, 'YNAM')
    
    remove_line(info, 'YUNI')
    line_AXS2_number=find_line(info, 'AXS2')
    info[line_AXS2_number]='AXS2    NONE\n'
    
    remove_line(info, 'A2RS')
    'Finding the line witth the title and adding "1D_modified"'
    line_TITL_number=find_line(info, 'TITL')
    addition='_1D_modified'
    info[line_TITL_number]=info[line_TITL_number][:-2]+addition+"'"+'\n'
    'Finding the line with the number of averages'
    line_AVGS_number=find_line(info, 'AVGS')
    averages1D = int(extract_value (info[line_AVGS_number]))
    'Total number of averages is the number of the good (not dumped) scans in the second dimensiion'
    'multiplied by the number of averages in each of them'
    total_averages=str(averages1D*scan_count)
    info[line_AVGS_number]=info[line_AVGS_number][:-2]+total_averages+'\n'
    
    '-----------------------------------------------------------'
    'saving the new DSC file'
    '-----------------------------------------------------------'
    text_file = open(save_name_DSC, 'w')
    text_file.writelines(info)
    text_file.close()

    
    
    