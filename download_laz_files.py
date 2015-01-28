'''
#==============================================================================
This script wil download the files which are listed in the downloadlist file
The .laz files will outamtically be unzipped. If for some reason the Internet
connection is broken, the script will stop running

Only input required is the right foldername (line 22)
#==============================================================================
'''

# import libraries
import numpy as np
import urllib2
import zipfile
import StringIO
import glob

#==============================================================================
# Input:
mainfolder = r'D:\rongen\Documents\External node data\\'
#==============================================================================

# Get the filelist
filelist = np.loadtxt(mainfolder+r'\filelist.txt', dtype = 'str')

# Check which laz files are already downloaded
filesinfolder = []
for name in glob.glob(mainfolder+r'\LAZfiles\*.laz'):
        tag = name.split('\\')[-1]
        filesinfolder.append(tag)

filesinfolder = np.array(filesinfolder)

# Download files
for i in filelist:

    # check if file is already in directory
    filetag = i.split('/')[-1]
    if sum(filesinfolder == filetag[:-4]) > 0:
        print(str(filetag[:-4])+' already in directory!')
        continue
    print('Downloading '+filetag)

    # Open url
    u = urllib2.urlopen(i)

    # read zip data
    zippedData = u.read()

    # Create temp file to store zipdata on
    outputFilename = mainfolder+'\\'+filetag
    temp = StringIO.StringIO()
    temp.write(zippedData)

    # extract data
    zipobj = zipfile.ZipFile(temp)
    uncompressed = zipobj.read(zipobj.namelist()[0])

    # save uncompressed data to disk
    output = open(outputFilename[:-4], 'wb')
    output.write(uncompressed)
    output.close()