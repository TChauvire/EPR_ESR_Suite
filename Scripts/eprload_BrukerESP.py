import re
import os
from warnings import warn
import numpy as np

def eprload_BrukerESP(fullbasename=None,Scaling=None,*args,**kwargs):
    '''
    ESP data file processing
    (Bruker EPR Standard for Spectrum Storage and Transfer)
    .par: descriptor file
    .spc: data file
    
    Bruker ECS machines
    Bruker ESP machines
    Bruker WinEPR, Simfonia
 
    This script is freely inspired by the easyspin suite from the Stefan Stoll lab
    (https://github.com/StollLab/EasySpin/)
    (https://easyspin.org/easyspin/)
    and from the pyspecdata python module from the John M. Franck lab (especially for the load_winepr_param function below)
    (https://github.com/jmfrancklab/pyspecdata)
    (https://pypi.org/project/pySpecData/)
    (http://jmfrancklab.github.io/pyspecdata/)
    
    Script written by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020
    
    Parameters
    ----------
    fullbasename : complete path to import the file, type is string
        DESCRIPTION. The default is None.
    Scaling : Scaling to achieve on the datafiles.
        DESCRIPTION:
        Different options are available: 'n', intensity is divided by the number of scans done
        'G', intensity is divided by the receiver gain                               
        'c', intensity is divided by the sampling time in second
        'P', intensity is divided by the microwave power in Watt
        'T', intensity is divided by the Temperature in Kelvin
        The default is None.

    Returns
    -------
    newdata : datafiles in numpy data array format. If newdata is a 1D datafile, the shape of the data will be (nx,1)
        
    abscissa : different abscissa associated with newdata.
        DESCRIPTION: If newdata is a 1D datafile, abscissa will be a column vector. 
                    If newdata is a 2D datafile, abscissa will be a two columns vector, the first column associated to the first dimension abscissa,
                    the second column associated to the 2nd dimension abscissa.
                    If newdata is a 3D datafile, abscissa will be a three columns vector, the first column associated to the first dimension abscissa,
                    the second column associated to the 2nd dimension abscissa,
                    the third column associated to the 3nd dimension abscissa.

    parameters : dictionnaries of the parameters reported in the .DSC bruker file.
    '''
    # Read parameter file (contains key-value pairs)
    filename = fullbasename[:-4]
    fileextension = fullbasename[-4:].lower()# case insensitive extension    
    if fileextension in ['.par','.spc']:
        filename_par = filename+'.par'
        filename_spc = filename+'.spc'
    else:
        raise ValueError("When guessing that the filename is a xepr file, the extension must be either .spc or .par\n"
                "This one is called".format(str(fullbasename)))

    if not os.path.exists(filename_spc):
        filename_spc = filename_spc[:-4] + filename_spc[-4:].upper()
        filename_par = filename_par[:-4] + filename_par[-4:].upper()
    
    parameters=load_winepr_param(filename_par)
    # FileType: flag for specific file format (outdated???)
    # w   Windows machines, WinEPR
    # c   ESP machines, cw EPR data
    # p   ESP machines, pulse EPR data
    FileType='c'

    twoD=0
    iscomplex=0
    nx=int(1024)
    ny=int(1)
    # Analyse data type flags stored in JSS.
    if 'JSS' in parameters:
        if len(bin(parameters['JSS'])) >= 13:
            if int(bin(parameters['JSS'])[-13]) == 1:
                twoD=1
        elif len(bin(parameters['JSS'])) >= 5:
            if int(bin(parameters['JSS'])[-5]) == 1:
                iscomplex=1

    # If present, SSX contains the number of x points.
    
    if 'SSX' in parameters:
        if twoD:
            if FileType == 'c':
                FileType='p'
            nx=int(parameters['SSX'])
            if iscomplex:
                nx=int(nx/2)
    
    # If present, SSY contains the number of y points.
    if 'SSY' in parameters:
        if twoD:
            if FileType == 'c':
                FileType='p'
            ny= int(parameters['SSY'])
    
    # If present, ANZ contains the total number of points.
    if 'ANZ' in parameters:
        nAnz=int(parameters['ANZ'])
        if twoD == 0:
            if FileType == 'c':
                FileType='p'
                nx=nAnz
            if iscomplex:
                nx=int(nx/2)
        else:
            if nx*ny != nAnz:
                raise ValueError('Two-dimensional data: SSX, SSY and ANZ in .par file are inconsistent.')
    
    # If present, RES contains the number of x points.
    if 'RES' in parameters:
        nx=int(parameters['RES'])
    
    # If present, REY contains the number of y points.
    if 'REY' in parameters:
        ny=int(parameters['REY'])
    
    # If present, XPLS contains the number of x points.
    if 'XPLS' in parameters:
        nx=int(parameters['XPLS'])
    
    # Preallocation of the data
    data = np.full((nx,ny),np.nan)
    maxlen = max(nx,ny)
    abscissa = np.full((maxlen,2),np.nan)
  
    # Construct abscissa vector
    if nx > 1:
        if 'JEX' not in parameters:
            parameters['JEX'] = 'field-sweep'
        if 'JEY' not in parameters:             
            parameters['JEY'] = ''
        # First abscissa creation: Name in order: 'GSI'/'GST' then 'GST'/'HCF' then 'XXLB'/'XXWI' then for Hiscore 'XXWI'/'SSX'/'??? Not sure for the step'
    if parameters['JEX'] == 'TimeSweep': #(Take GH = ???)
        if 'RCT' in parameters:
            ConversionTime=float(parameters['RCT'])
        else:
            ConversionTime=1
        nstop = (nx-1)*(ConversionTime)/1000.0
        abscissa[:,0]=np.linspace(0,nstop,nx-1)           
    elif parameters['JEX'] == 'ENDOR': #(Take GH = 1)
        GST = float(parameters['GST'])
        GSI = float(parameters['GSI'])
        abscissa[:,0]= GST+GSI*(np.linspace(0,1,nx))
    elif parameters['JEX'] == 'field-sweep': 
        if 'GSI' in parameters:
            GST = float(parameters['GST'])
            GSI = float(parameters['GSI'])
            abscissa[:,0]= GST+GSI*(np.linspace(0,1,nx))
        elif 'XXLB' in parameters:
            XXLB = float(parameters['XXLB'])
            XXWI = float(parameters['XXWI'])
            abscissa[:,0] = XXLB + np.linspace(0,XXWI,nx)        
        elif 'HCF' in parameters:
            HCF = float(parameters['HCF'])
            GST = float(parameters['GST'])
            HSW = float((HCF-GST)*2)
            parameters['HSW'] = HSW
            abscissa[:,0]=HCF+HSW/2*np.linspace(-1,1,nx)
    else: 
        raise ValueError('Parameter file is corrupted. Starting field value can\'t be determined')
    
    # Second abscissa creation: 'JEY'

    #if parameters['JEY'] == 'temp-sweep': #(Take GH = 2)
    #    HCF = float(parameters['HCF'])
    #    HSW = float(parameters['HSW'])
    #    abscissa[:ny,1]=HCF+HSW/2*np.linspace(-1,1,nx)
    if twoD: # 2D dataset
        if 'XYLB'in parameters:
            XYLB = float(parameters['XYLB'])
            XYWI = float(parameters['XYWI'])
            abscissa[:ny,1]= XYLB + np.linspace(0,XYWI,ny)
    else:
        warn('Could not determine second abscissa range from parameter file!')
        ny = 1
    # In case of column filled with Nan, erase the column in the array
    abscissa = abscissa[:,~np.isnan(abscissa).all(axis=0)]
    #Check for byte type endian
    if 'DOS' in parameters:
        FileType = 'w'

    # For DOS ByteOrder is ieee-le, in all other cases ieee-be
    # Set number format
    if 'c' == FileType:
        dt_spc=np.dtype('int32')
    elif 'w' == FileType:
        dt_spc=np.dtype('float32')
    elif 'p' == FileType:
        dt_spc=np.dtype('int32')
    dt_data = dt_spc
    if 'DOS' in parameters: # It's little Endian format
        dt_spc = dt_spc.newbyteorder('<')   
    else: # It's big Endian format
        dt_spc = dt_spc.newbyteorder('>')
    # Read data file.  
    with open(filename_spc,'rb') as fp:
        data = np.frombuffer(fp.read(),dtype=dt_spc)
    newdata = np.copy(data.reshape(ny,nx).T)
    
    if iscomplex == 1:
        newdata = newdata.astype(dtype=dt_data).view(dtype=np.complex).reshape(nx,ny)    
 
    # Scale spectrum/spectra
    if Scaling == None:
        pass
    else:
        # Number of scans1
        if Scaling == 'n':
            if 'JSD' not in parameters:
                warn('Cannot scale by number of scans, since JSD is absent in parameter file.')
            else:
                nScansDone=float(parameters['JSD'])
                newdata=newdata / nScansDone
        # Receiver gain
        if Scaling == 'G':
            if 'RRG' not in parameters:
                #parameters.RRG = '2e4'; # default value on UC Davis ECS106
                warn('Cannot scale by gain, since RRG is absent in parameter file.')
            else:
                ReceiverGain=float(parameters['RRG'])
                newdata=newdata / ReceiverGain
        # Microwave power111
        if Scaling == 'P':
            if 'MP' not in parameters:
                warn('Cannot scale by power, since MP is absent in parameter file.')
            elif ~twoD:
                mwPower=float(parameters['MP'])
                newdata=newdata / np.sqrt(mwPower)
            else:
                # 2D power sweep, power along second dimension
                XYLB = float(parameters['XYLB'])
                XYWI = float(parameters['XYWI'])
                dB=XYLB + np.linspace(0,XYWI,ny)
                mwPower=float(parameters['MP'])
                abscissa[:ny,1]=mwPower*(10.0**(-1*dB/10))
                for iPower in range(ny):
                    newdata[:,iPower]=newdata[:,iPower]/np.sqrt(abscissa[iPower,1])
        # Temperature
        if Scaling == 'T':
            if 'TE' not in parameters:
                warn('Cannot scale by temperature, since TE is absent in parameter file.')
            else:
                Temperature=float(parameters['TE'])
                if Temperature == 0:
                    warn('Cannot scale by temperature, since TE is zero in parameter file.')
                else:
                    newdata=newdata*Temperature
        # Conversion/sampling time
        if Scaling == 'c':
            if 'RCT' not in parameters:
                warn('Cannot scale by sampling time, since RCT in the .par file is missing.')
            else:
                RCT=float(parameters['RCT'])
                newdata=newdata/RCT
  
    return newdata,abscissa,parameters
    
def load_winepr_param(filename_par):
    '''
    Load the parameters for the winepr filename, which should be a .par/.PAR extension. 
    
    This script is freely inspired by the pyspecdata python module from the John M. Franck lab (especially for the xepr_load_acqu function below)
    (https://github.com/jmfrancklab/pyspecdata)
    (https://pypi.org/project/pySpecData/)
    (http://jmfrancklab.github.io/pyspecdata/)
    
    Script adapted by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020

    Returns
    -------
    A dictionary of the parameter written in the .par file.
    '''
    with open(filename_par,'r') as fp:
        lines = fp.readlines()
    line_re = re.compile(r'([_A-Za-z0-9]+) +(.*)')
    lines = map(str.rstrip,lines)
    lines = [j.rstrip('\n') for j in lines] # because it's just \n, even on windows
    v = {'HSW':50} # needed for some general import features... ('DRS':4096,'RES':1024,)
    for line in lines:
        m = line_re.match(line)
        if m is None:
            warn('Warning: {0} does not appear to be a valid'
                               'WinEPR format line, and I suspect this is'
                               ' a problem with the terminators!'.format(str(line)))
        else:
            name = m.groups()[0]
            value = m.groups()[1]
            try:
                value = int(value)
            except:
                try:
                    value = float(value)
                except:
                    pass
            v[name]=value
    return v