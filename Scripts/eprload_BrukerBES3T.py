from warnings import warn
import re
import numpy as np
import os
from io import StringIO
# -------------------------------------------------------------------------------


def eprload_BrukerBES3T(fullbasename=None, Scaling=None, *args, **kwargs):
    '''
    BES3T file processing
    (Bruker EPR Standard for Spectrum Storage and Transfer)
    .DSC: descriptor file
    .DTA: data file
    used on Bruker ELEXSYS and EMX machines Code based on BES3T version 1.2 (Xepr 2.1)

    This script is freely inspired by the easyspin suite from the Stefan Stoll lab
    (https://github.com/StollLab/EasySpin/)
    (https://easyspin.org/easyspin/)
    and from the pyspecdata python module from the John M. Franck lab (especially for the xepr_load_acqu function below)
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

    abscissa : different abscissa associated with newdata, numpy data array format.
        DESCRIPTION: If newdata is a 1D datafile, abscissa will be a column vector. 
                    If newdata is a 2D datafile, abscissa will be a two columns vector, the first column associated to the first dimension abscissa,
                    the second column associated to the 2nd dimension abscissa.
                    If newdata is a 3D datafile, abscissa will be a three columns vector, the first column associated to the first dimension abscissa,
                    the second column associated to the 2nd dimension abscissa,
                    the third column associated to the 3nd dimension abscissa.

    parameters : dictionnaries of the parameters reported in the .DSC bruker file.
    '''
    nx = 1024
    ny = 1
    nz = 1

    filename = fullbasename[:-4]
    fileextension = fullbasename[-4:].upper()  # case insensitive extension
    if fileextension in ['.DSC', '.DTA']:
        filename_par = filename+'.DSC'
        filename_spc = filename+'.DTA'
    else:
        raise ValueError("When guessing that the filename is a xepr file, the extension must be either .DSC or .DAT\n"
                         "This one is called {0}".format(str(fullbasename)))

    if not os.path.exists(filename_spc):
        filename_spc = filename_spc[:-4] + filename_spc[-4:].lower()
        filename_par = filename_par[:-4] + filename_par[-4:].lower()

    # Read descriptor file (contains key-value pairs)
    parameters = xepr_load_acqu(filename_par)

    # XPTS: X Points   YPTS: Y Points   ZPTS: Z Points
    # XPTS, YPTS, ZPTS specify the number of data points in
    # x, y and z dimension.
    if 'XPTS' in parameters:
        nx = int(parameters['XPTS'])
    else:
        raise ValueError('No XPTS in DSC file.')

    if 'YPTS' in parameters:
        ny = int(parameters['YPTS'])
    else:
        ny = 1
    if 'ZPTS' in parameters:
        nz = int(parameters['ZPTS'])
    else:
        nz = 1
    # BSEQ: Byte Sequence
    # BSEQ describes the byte order of the data. BIG means big-endian (encoding = '>'), LIT means little-endian (encoding = '<'). Sun and Motorola-based systems are
    # big-endian (MSB first), Intel-based system little-endian (LSB first).
    if 'BSEQ' in parameters:
        if 'BIG' == parameters['BSEQ']:
            byteorder = '>'
        elif 'LIT' == parameters['BSEQ']:
            byteorder = '<'
        else:
            raise ValueError('Unknown value for keyword BSEQ in .DSC file!')
    else:
        warn('Keyword BSEQ not found in .DSC file! Assuming BSEQ=BIG.')
        byteorder = '>'

    # IRFMT: Item Real Format
    # IIFMT: Item Imaginary Format
    # Data format tag of BES3T is IRFMT for the real part and IIFMT
    # for the imaginary part.
    if 'IRFMT' in parameters:
        if 'C' == parameters['IRFMT'].upper():
            dt_spc = np.dtype('int8')
        elif 'S' == parameters['IRFMT'].upper():
            dt_spc = np.dtype('int16')
        elif 'I' == parameters['IRFMT'].upper():
            dt_spc = np.dtype('int32')
        elif 'F' == parameters['IRFMT'].upper():
            dt_spc = np.dtype('float32')
        elif 'D' == parameters['IRFMT'].upper():
            dt_spc = np.dtype('float64')
        elif 'A' == parameters['IRFMT'].upper():
            raise TypeError('Cannot read BES3T data in ASCII format!')
        elif ('0' or 'N') == parameters['IRFMT'].upper():
            raise ValueError('No BES3T data!')
        else:
            raise ValueError('Unknown value for keyword IRFMT in .DSC file!')
    else:
        raise ValueError('Keyword IRFMT not found in .DSC file!')

    # We enforce IRFMT and IIFMT to be identical.
    if 'IIFMT' in parameters:
        if parameters['IIFMT'].upper() != parameters['IRFMT'].upper():
            raise ValueError('IRFMT and IIFMT in DSC file must be identical.')

    # Preallocation of theabscissa
    maxlen = max(nx, ny, nz)
    abscissa = np.full((maxlen, 3), np.nan)
    # Construct abscissa vectors
    AxisNames = ['X', 'Y', 'Z']
    Dimensions = [nx, ny, nz]
    for a in AxisNames:
        index = AxisNames.index(a)
        axistype = parameters[str(a+'TYP')]
        if Dimensions[index] == 1:
            pass
        else:
            if 'IGD' == axistype:
                # Nonlinear axis -> Try to read companion file (.XGF, .YGF, .ZGF)
                companionfilename = str(filename+'.'+a+'GF')
                if 'D' == parameters[str(a+'FMT')]:
                    dt_axis = np.dtype('float64')
                elif 'F' == parameters[str(a+'FMT')]:
                    dt_axis = np.dtype('float32')
                elif 'I' == parameters[str(a+'FMT')]:
                    dt_axis = np.dtype('int32')
                elif 'S' == parameters[str(a+'FMT')]:
                    dt_axis = np.dtype('int16')
                else:
                    raise ValueError('Cannot read data format {0} for companion file {1}'.format(
                        str(a+'FMT'), companionfilename))
                dt_axis = dt_axis.newbyteorder(byteorder)
                # Open and read companion file
                try:
                    with open(companionfilename, 'rb') as fp:
                        abscissa[:Dimensions[index], index] = np.frombuffer(
                            fp.read(), dtype=dt_axis)
                except:
                    warn(
                        f'Could not read companion file {companionfilename} for nonlinear axis. Assuming linear axis.')
                axistype = 'IDX'
        if axistype == 'IDX':
            minimum = float(parameters[str(a+'MIN')])
            width = float(parameters[str(a+'WID')])
            npts = int(parameters[str(a+'PTS')])
            if width == 0:
                warn('Warning: {0} range has zero width.\n'.format(
                    AxisNames[a]))
                minimum = 1.0
                width = len(a) - 1.0
            abscissa[:Dimensions[index], index] = np.linspace(
                minimum, minimum+width, npts)
        if axistype == 'NTUP':
            raise ValueError('Cannot read data with NTUP axes.')
    # In case of column filled with Nan, erase the column in the array
    abscissa = abscissa[:, ~np.isnan(abscissa).all(axis=0)]
    dt_data = dt_spc
    dt_spc = dt_spc.newbyteorder(byteorder)

    # Read data matrix and separate complex case from real case.
    newdata = np.full((nx, ny, nz), np.nan)
    # reorganize the data in a "complex" way as the real part and the imaginary part are separated
    # IKKF: Complex-data Flag
    # CPLX indicates complex data, REAL indicates real data.
    if 'IKKF' in parameters:
        if parameters['IKKF'] == 'REAL':
            data = np.full((nx, ny, nz), np.nan)
            with open(filename_spc, 'rb') as fp:
                data = np.frombuffer(fp.read(), dtype=dt_spc)
                # data = np.frombuffer(
                #     fp.read(), dtype=dt_spc).reshape(nx, ny, nz)
            newdata = np.copy(data)
        elif parameters['IKKF'] == 'CPLX':
            dt_new = np.dtype('complex')
            with open(filename_spc, 'rb') as fp:
                data = np.frombuffer(fp.read(), dtype=dt_spc)
                # Check if there is multiple harmonics (High field ESR quadrature detection)
                # outer dimension for the 90 degree phase
                harmonics = np.array([[False] * 5]*2)
                for j, jval in enumerate(['1st', '2nd', '3rd', '4th', '5th']):
                    for k, kval in enumerate(['', '90']):
                        thiskey = 'Enable'+jval+'Harm'+kval
                        if thiskey in parameters.keys() and parameters[thiskey]:
                            harmonics[k, j] = True
                n_harmonics = sum(harmonics)[0]
                if n_harmonics != 0:
                    ny = int(len(data)/nx/n_harmonics)

            # copy the data to a writable numpy array
            newdata = np.copy(data.astype(dtype=dt_data).view(dtype=dt_new))
        else:
            raise ValueError('Unknown value for keyword IKKF in .DSC file!')
    else:
        warn('Keyword IKKF not found in .DSC file! Assuming IKKF=REAL.')
    # Split 1D-array into 3D-array according to XPTS/YPTS/ZPTS
    newdata = np.array_split(newdata, nz)
    newdata = np.array(newdata).T
    newdata = np.array_split(newdata, ny)
    newdata = np.array(newdata).T
    # Ensue proper numpy formatting
    newdata = np.atleast_1d(newdata)
    newdata = np.squeeze(newdata)

    # # Abscissa formatting
    # abscissa = np.atleast_1d(abscissa)
    # abscissa = np.squeeze(abscissa)
    if Scaling == None:
        pass
    else:
        # Averaging over the number of scans
        if Scaling == 'n':
            # Check if the experiments was already averaged
            # Number of scans
            if 'AVGS' in parameters:
                nAverages = float(parameters['AVGS'])
                if 'SctNorm' in parameters:
                    if parameters['SctNorm'] == True:
                        warn(
                            'Scaling by number of scans not possible,\nsince data in DSC/DTA are already averaged')
                else:
                    warn(
                        "Missing SctNorm field in the DSC file. Cannot determine whether data is already scaled, assuming it isn't...")
                    newdata = newdata/nAverages
            else:
                raise warn('Missing AVGS field in the DSC file.')

        # Receiver gain
        if parameters['EXPT'] == 'CW' and Scaling == 'G':
            # SPL/RCAG: Receiver Gain in dB
            if 'RCAG' in parameters:
                ReceiverGaindB = float(parameters['RCAG'])
                ReceiverGain = 10 ** (ReceiverGaindB / 20)
                newdata = newdata/ReceiverGain
            else:
                warn(
                    'Cannot scale by receiver gain, since RCAG in the DSC file is missing.')

        # Conversion/sampling time
        elif parameters['EXPT'] == 'CW' and Scaling == 'c':
            # SPL/SPTP: sampling time in seconds
            if 'STPT' in parameters:
                # Xenon (according to Feb 2011 manual) already scaled data by ConvTime if
                # normalization is specified (SctNorm=True). Question: which units are used?
                # Xepr (2.6b.2) scales by conversion time even if data normalization is
                # switched off!
                ConversionTime = float(parameters['SPTP'])
                newdata = newdata/(ConversionTime*1000)
            else:
                warn(
                    'Cannot scale by sampling time, since SPTP in the DSC file is missing.')
        # Microwave power
        elif parameters['EXPT'] == 'CW' and Scaling == 'P':
            # SPL/MWPW: microwave power in watt
            if 'MWPW' in parameters:
                mwPower = float(parameters['MWPW'])*1000
                newdata = newdata/np.sqrt(mwPower)
            else:
                warn('Cannot scale by power, since MWPW is absent in parameter file.')
        # else:
        #    warn('Cannot scale by microwave power, since these are not CW-EPR data.')

        # Temperature
        if Scaling == 'T':
            # SPL/STMP: temperature in kelvin
            if 'STMP' in parameters:
                Temperature = float(parameters['STMP'])
                newdata = newdata*Temperature
            else:
                warn('Cannot scale by temperature, since STMP in the DSC file is missing.')

    return newdata, abscissa, parameters


def xepr_load_acqu(filename_par=None):
    '''
    Load the Xepr acquisition parameter file, which should be a .dsc/.DSC extension.

    This script is freely inspired by the pyspecdata python module from the John M. Franck lab (especially for the xepr_load_acqu function below)
    (https://github.com/jmfrancklab/pyspecdata)
    (https://pypi.org/project/pySpecData/)
    (http://jmfrancklab.github.io/pyspecdata/)

    Script adapted by Timothée Chauviré (https://github.com/TChauvire/EPR_ESR_Suite/), 09/09/2020

    Returns
    -------
    A dictionary of the parameter written in the .DSC file.
    Because of the format of the .DSC files, this is a dictionary of
    dictionaries, where the top-level keys are the hash-block (*i.e.*
    ``#DESC``, *etc.*).
    '''
    def auto_string_convert(x):
        '''genfromtxt is from numpy -- with dtype=None, it does
        automatic type conversion -- note that strings with
        spaces will be returned as a record array it appears to
        need this StringIO function rather than a string because
        it's designed to read directly from a file.  The tolist
        converts the record array to a list.'''
        if len(x):
            try:
                return np.genfromtxt(StringIO(x), dtype=None, encoding='str').tolist()
            except:
                raise ValueError("genfromtxt chokes on "+repr(x))
        else:
            return None

    which_block = None
    block_list = None
    block_re = re.compile(r'^ *#(\w+)')
    comment_re = re.compile(r'^ *\*')
    variable_re = re.compile(r'^ *([^\s]*)\s+(.*?) *$')
    comma_re = re.compile(r'\s*,\s*')
    with open(filename_par, 'r') as fp:
        blocks = {}
        # read lines and assign to the appropriate block
        for line in fp:
            m = comment_re.search(line)
            if m:
                pass
            else:
                m = block_re.search(line)
                if m:
                    if which_block is not None:
                        blocks.update({which_block: dict(block_list)})
                    which_block = m.groups()[0]
                    block_list = []
                else:
                    if which_block is None:
                        raise ValueError(
                            "Appears to be stuff outside the first hashed block which, as far as I know, should not be allowed.  The first non-comment line I see is: "+repr(line))
                    else:
                        m = variable_re.search(line)
                        if m:
                            if ',' in m.groups()[1]:
                                # break into lists
                                block_list.append((m.groups()[0],
                                                   map(auto_string_convert,
                                                       comma_re.split(
                                                           m.groups()[1]))))

                            else:
                                block_list.append((m.groups()[0],
                                                   auto_string_convert(
                                    m.groups()[1])))
                        else:
                            raise ValueError(
                                "I don't know what to do with the line:\n"+line)

        blocks.update({which_block: dict(block_list)})
    # flatten the dictionary
    parameters = {}
    for k_a, v_a in blocks.items():
        parameters.update(v_a)
    if 'IRFMT' in parameters:
        if type(parameters['IRFMT']) == map:
            a = str(list(parameters['IRFMT'])[0])
            parameters['IRFMT'] = a
    if 'IIFMT' in parameters:
        if type(parameters['IIFMT']) == map:
            a = str(list(parameters['IIFMT'])[0])
            parameters['IIFMT'] = a
    if 'IKKF' in parameters:
        if type(parameters['IKKF']) == map:
            a = str(list(parameters['IKKF'])[0])
            parameters['IKKF'] = a
    return parameters
