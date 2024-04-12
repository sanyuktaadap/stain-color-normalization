#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------------
# Name          : Get Stain Histograms
# Date          : Dec 3, 2023
# Author        : Jose L. Agraz, PhD
# Co-Author     : Caleb Grenko
# Funding       : Spyros Bakas, PhD
#
# Description   : This program aggregates the stain vectors and histograms
#                 in a cohort. The result is stored in a dataframe file
#
# Usage Example :
#   python Aggregate_Stain_Vectors_and_Histograms.py\
#             --Histogram_Dataframe_Directory     /cbica/home/agrazj/Normalization/BaSSaN-Update/Produce_Dataframes/Output/1864_IvyGAP_Images/Dataframes
#             --Stain_Vector_Dataframe_Directory  /cbica/home/agrazj/Normalization/BaSSaN-Update/Produce_Single_Image_Normalization/Dataframes/Full_Cohort
#             --Output_Directory                  /cbica/home/agrazj/Normalization/BaSSaN-Update/Produce_Single_Image_Normalization/
#             --Number_of_Images                  1864
#       
# Notes:
#
# ------------------------------------------------------------------------------------------------
# Library imports
# ------------------------------------------------------------------------------------------------
import sys
import glob
import random
import shutil
import argparse
import numpy             as np
import pandas            as pd
import dask.dataframe    as dd
from pathlib             import Path
from datetime            import datetime
#
__author__  = ['Jose L. Agraz, PhD']
__status__  = "Public_Access"
__credits__ = ['Spyros Bakas','Caleb Grenko']
__version__ = "0.0.1"
#-----------------------------------------------------------------
# Constants
#-----------------------------------------------------------------
MEMORY_DATA_ENGINE                          = 'pyarrow'
RED_COLOR                                   = 0
GREEN_COLOR                                 = 1
BLUE_COLOR                                  = 2
#------------------------------------------------------------------------------------------------
# Function Name: Get Arguments
# Author: Jose L. Agraz, PhD
# Date: 03/12/2020
# Description: Define input arguments using flags
# Input: Slides, Label Map Color file, and output file
# Output: Argument list
#------------------------------------------------------------------------------------------------
def GetArguments():
    DESCRITPTION_MESSAGE = \
    'Description: This program adds histograms in H&E results in dataframe files.       \n' +\
    '             The script calculates the optimal bin size for every feature          \n' +\
    '             Then, selects the largest optimal bin size and calculates the         \n' +\
    '             histogram for every feature using these bins. Finally, the            \n' +\
    '             scripts merges all histograms and stores the info in a npy file       \n' +\
    '             with format: [[Hematoxylin,Bin],[Eosin,bin]]                          \n' +\
    '                                                                                   \n' +\
    'usage:                                                                             \n' +\
    'Aggregate_Stain_Vectors_and_Histograms.py                                          \n' +\
    '     --Histogram_Dataframe_Directory     Histogram_Dataframes                      \n' +\
    '     --Stain_Vector_Dataframe_Directory  Stain_Vectors_Dataframes                  \n' +\
    '     --Output_Directory                  Normalization_Parameters_Directory        \n' +\
    '     --Number_of_Images                  1864                                        '
     

    parser = argparse.ArgumentParser(description=DESCRITPTION_MESSAGE)

    parser.add_argument('-m', '--Histogram_Dataframe_Directory',    required=True, help='Histogram Dataframe Directory')
    parser.add_argument('-s', '--Stain_Vector_Dataframe_Directory', required=True, help='Stain Vector Dataframe Directory')
    parser.add_argument('-o', '--Output_Directory',                 required=True, help='Output Directory')
    parser.add_argument('-i', '--Number_of_Images',                 required=True, help='Number of Random Images')
    
    parser.add_argument('-v', '--version', action='version', version= "%(prog)s (ver: "+__version__+")")    
        
    args = parser.parse_args()
    
    return args

#-----------------------------------------------------------------
# Function Name: Terminate
# Author: Jose L. Agraz, PhD 
# Date: 04/14/2020
# Description: Wrap up housekeeping
# Input: Histograms
# Output:
#-----------------------------------------------------------------
def Terminate(ResultsDirectory,HistogramData,StainVectorsData):    
    global InputArguments
    
    if len(HistogramData) and len(StainVectorsData):
        FileName      = InputArguments.Number_of_Images + 'ImageCohortHistograms'
        np.save(str(Path(ResultsDirectory) / FileName), HistogramData) 
    
        FileName      = InputArguments.Number_of_Images + 'ImageCohortStainVectors'
        np.save(str(Path(ResultsDirectory) / FileName), StainVectorsData) 
                    
        print('* Dataframe Files Available    : {}                '.format(InputArguments.Number_of_Images))
        print('* Histograms Directory Name    : {}                '.format(InputArguments.Histogram_Dataframe_Directory))
        print('* Stain Vectors Directory Name : {}                '.format(InputArguments.Stain_Vector_Dataframe_Directory))
        print('* Results Directory Name       : {}                '.format(ResultsDirectory))
        
        if len(CohortFileList) == 1:
            Source          = CohortFileList[0]
            Image_File_Name = str(Path(CohortFileList[0]).name)
            Destination     = str(Path(ResultsDirectory) / Image_File_Name)
            print('* Single Image File Name   : {}                   '.format(Image_File_Name))
            shutil.copyfile(Source,Destination)            
    else:
        print('Empty Histograms and/or Stain Vectors')
         
# ------------------------------------------------------------------------------------------------
# Function Name: Creates a directory
# Author: Jose L. Agraz, PhD
# Date: 04/12/2020
# Description: Created a directory
# Input: path
# Output: none
# ------------------------------------------------------------------------------------------------
def CreateDirectory(OutputPath):
    try:
        Path(OutputPath).mkdir(parents=True, exist_ok=True)
    except:
        print('Could not created directory:\n{}'.format(OutputPath))
        sys.exit()    
#-----------------------------------------------------------------
# Function Name: Initialize
# Author: Jose L. Agraz, PhD 
# Date: 04/14/2020
# Description: Sets up input and housekeeping
# Input: None
# Output: File list
#-----------------------------------------------------------------
def Initialize():
    global InputArguments
    OUTPUT_FILE_EXTENSION      = 'parquet'
    SourcePath                 = str()
    DataframeFiles             = list()
    NumberOfAvailableFiles     = int()
    NumberOfRequestedFiles     = int()
    #-----------------------------------------------------------------
    print('Initializing...')
    print('Creating Directories')
    ResultsDirectory = InputArguments.Number_of_Images + '_Image_Cohort_Aggregated_Normalization_Parameters'
    OutputDirectory  = str(Path(InputArguments.Output_Directory) / ResultsDirectory)
    #CreateDirectory(OutputDirectory)
    #-----------------------------------------------------------------
    SourcePath     = str(Path(InputArguments.Histogram_Dataframe_Directory) / str('*.' + OUTPUT_FILE_EXTENSION))
    print('Source Path: {}'.format(SourcePath))
    print('Build a list of dataframe files')
    DataframeFiles = glob.glob(SourcePath)
    #-----------------------------------------------------------------    

    NumberOfAvailableFiles  = len(DataframeFiles)
    NumberOfRequestedFiles  = int(InputArguments.Number_of_Images)
    if NumberOfRequestedFiles <= NumberOfAvailableFiles :
        print('Requesting {} Cohorts from {} Dataframe Avaiable'.format(InputArguments.Number_of_Images,NumberOfAvailableFiles))        
    else:        
        raise IOError('Asked for more Images ({}), than available ({})'.format(InputArguments.Number_of_Images,NumberOfAvailableFiles))        

    return DataframeFiles,OutputDirectory
    
#-----------------------------------------------------------------
# Function Name: Unravel Array
# Author: Jose L. Agraz, PhD 
# Date: 04/14/2020
# Description: Expands the pixel color and pixel count arrays
#              to a single array to calculate histogram
#              Code uses list comprehension to speed up flow
#              450395.69 ms
# Input: pixel color and pixel count arrays
# Output: flat array
#-----------------------------------------------------------------
def ImprovedUnravelArray(PixelColorArray,PixelCountArray):
    
    MyList = list()
    for PixelColor,PixelCount in zip(PixelColorArray,PixelCountArray):
        MyList += int(PixelCount)*[PixelColor]

    del PixelColorArray,PixelCountArray    
    return MyList

#-----------------------------------------------------------------
# Function Name: Find Stain Vectors
# Author: Jose L. Agraz, PhD 
# Date: 04/14/2020
# Description: Find total Area
# Input: File list
# Output: Total Area
#-----------------------------------------------------------------
def UpdateUserWithStainVectors(StainName,StainVector):
    # ---------------------------------------------------------------------------------
    print('****************************************')
    print('Stain Vectors Cohort Results            ')
    print('****************************************')
    print('Stain Name         :{}                  '.format(StainName))
    print('Stain Vectors:                          ')
    print('   Red             :{}                  '.format(StainVector[RED_COLOR]))
    print('   Green           :{}                  '.format(StainVector[GREEN_COLOR]))
    print('   Blue            :{}                  '.format(StainVector[BLUE_COLOR]))
    print('****************************************')  
         
#-----------------------------------------------------------------
# Function Name: Sum Histograms
# Author: Jose L. Agraz, PhD 
# Date: 04/14/2020
# Description: Find total Area
# Input: File list
# Output: Total Area
#-----------------------------------------------------------------
def SumHistograms(CohortFileList,StainName,BinSize):
    NORMALIZE_HISTOGRAM_FACTOR = 100000
    DaskDataFrameArray         = list()
    
    print('{}: Get Dataframe Cohort List'.format(StainName))
    DaskDataFrameArray   = [dd.read_parquet(SingleFile,engine=MEMORY_DATA_ENGINE,columns=[StainName]).repartition(partition_size="100MB") for SingleFile in CohortFileList]
    print('{}: Concatenate Dataframes'.format(StainName))
    DaskDataframe        = dd.concat(DaskDataFrameArray,axis = 1,ignore_index=True)    
    print('{}: Sum {} Histograms with {} Bins'.format(StainName,len(DaskDataframe.columns),len(DaskDataframe.index)))
    CohortHistogram      = DaskDataframe.sum(axis = 1, skipna = True)
    print('{}: Normalize from {} Histograms,TypeCast, and Convert Series to Numpy array'.format(StainName,len(CohortFileList)))
    print('Compute Dask Histogram')
    Sum_Cohort_Histogram = CohortHistogram.astype(np.uint32).values.compute()
    print('Normalize Histogram to {} Max Count'.format(NORMALIZE_HISTOGRAM_FACTOR))    
    CohortHistogram = NORMALIZE_HISTOGRAM_FACTOR * np.true_divide(Sum_Cohort_Histogram, max(Sum_Cohort_Histogram))
            
    del DaskDataFrameArray,DaskDataframe
    
    return CohortHistogram.astype(np.uint32)

#-----------------------------------------------------------------
# Function Name: Process Stain Vector
# Author: Jose L. Agraz, PhD 
# Date: 04/14/2020
# Description: 
# Input: 
# Output: 
#----------------------------------------------------------------- 
def ProcessStainVector(CohortFileList,StainName):
    global InputArguments

    FileList         = list()
    FilePath         = str()
    FileName         = str()
    StainVectorsList = [list(),list(),list()]   
    NumberOfRows     = int()
    TotalCohortArea  = int()
    
    GlobalStartTimer = datetime.now()
    print('Global Start Time (hh:mm:ss.ms) {}'.format(GlobalStartTimer))
    
    # ----------------------------------------------------------------------------
    for FilePath in CohortFileList:
        FileName = str(Path(FilePath).name)
        FileList += [str(Path(InputArguments.Stain_Vector_Dataframe_Directory) / FileName)]
    # ----------------------------------------------------------------------------
    print('{}: FileList:\n{}'.format(StainName,FileList))
    print('{}: Get First Dataframe Cohort List'.format(StainName))
    
    try:
        DaskDataFrameArray = [dd.read_parquet(SingleFile,engine=MEMORY_DATA_ENGINE).repartition(partition_size="100MB") for SingleFile in FileList]
    except:
       raise IOError('Dask read parquet Exception triggered!!!')    
       
    print('{}: Concatenate Dataframes'.format(StainName))
    DaskDataframe      = dd.concat(DaskDataFrameArray,ignore_index=True,axis = 0).repartition(partition_size="100MB")  
    print('{}: Extract Stain Vectors'.format(StainName))
    StainVectors       = DaskDataframe[['RGBStainVector_Red_W',\
                                        'RGBStainVector_Green_W',\
                                        'RGBStainVector_Blue_W',\
                                        'Area']][DaskDataframe.Stain == StainName.title()]      
        
    NumberOfRows = len(StainVectors.index)   
    if NumberOfRows:     
        print('{}: Computing Stain Vectors'.format(StainName))
        TotalCohortArea               = StainVectors.Area.sum(skipna = True).compute()
        print('{}: Total Cohort Area: {} Pixels'.format(StainName,TotalCohortArea))

        StainVectorsList[RED_COLOR]   = int(((StainVectors.Area * StainVectors.RGBStainVector_Red_W).sum()  /TotalCohortArea).compute())
        print('{}: Red Stain Vector: {}'.format(StainName,StainVectorsList[RED_COLOR]))
        
        StainVectorsList[GREEN_COLOR] = int(((StainVectors.Area * StainVectors.RGBStainVector_Green_W).sum()/TotalCohortArea).compute())
        print('{}: Green Stain Vector: {}'.format(StainName,StainVectorsList[GREEN_COLOR]))
        
        StainVectorsList[BLUE_COLOR]  = int(((StainVectors.Area * StainVectors.RGBStainVector_Blue_W).sum() /TotalCohortArea).compute())
        print('{}: Blue Stain Vector: {}'.format(StainName,StainVectorsList[BLUE_COLOR]))

    print('{}: Stain Vector: {}'.format(StainName,StainVectorsList))      
    
    del DaskDataFrameArray,DaskDataframe
    
    UpdateUserWithStainVectors(StainName,StainVectorsList)    
    
    return StainVectorsList

#-----------------------------------------------------------------
if __name__ == "__main__":
    
    HEMATOXYLIN_STAIN_LABEL = 'Hematoxylin'
    EOSIN_STAIN_LABEL       = 'Eosin'
    FIRST_ITEM              = 0
    CohortHistogramFileList = list()   
    CohortFileList          = list()
    StainName               = str()
    BinSize                 = int()
    HematoxylinHistogram    = list()
    EosinHistogram          = list()
    HematoxylinStainVectors = list()
    EosinStainVectors       = list()
    StainVectorsData        = list()
    HistogramData           = list()
    #-----------------------------------
    # Initialize arguments
    InputArguments          = GetArguments()    
    CohortHistogramFileList,\
    ResultsDirectory        = Initialize()
    
    print('Fetching Number of Bins from Dataframe Columns')
    DataframeDataBase       = pd.read_parquet(CohortHistogramFileList[FIRST_ITEM], engine=MEMORY_DATA_ENGINE)   
    BinSize                 = len(DataframeDataBase.index)+1
    BinsLocations           = np.arange(BinSize-1)                       
    #-----------------------------------------------------------------                
    CohortFileList          = random.sample(CohortHistogramFileList, k=int(InputArguments.Number_of_Images))
    #-----------------------------------------------------------------                
    StainName               = HEMATOXYLIN_STAIN_LABEL.lower()
    print('{}: Calculating Histogram'.format(StainName))
    try:       
        HematoxylinHistogram    = SumHistograms(CohortFileList, StainName,BinSize)
    except:
       raise IOError('Histograms Sum Exception triggered!!!')
    #-----------------------------------------------------------------                
    print('{}: Calculating Stain Vectors'.format(StainName))
    HematoxylinStainVectors = ProcessStainVector(CohortFileList,StainName)
    #-----------------------------------------------------------------                
    StainName               = EOSIN_STAIN_LABEL.lower()
    print('{}: Calculating Histogram'.format(StainName))
    try:
        EosinHistogram          = SumHistograms(CohortFileList, StainName,BinSize)
    except:
       raise IOError('Stain Vectors Sum Exception triggered!!!')          
    #-----------------------------------------------------------------                       
    print('{}: Calculating Stain Vectors'.format(StainName))
    try:    
        EosinStainVectors       = ProcessStainVector(CohortFileList,StainName)        
    except:
       raise IOError('Process Stain Vectors Exception triggered!!!')                  
    #-----------------------------------------------------------------                
    HistogramData           = [[HematoxylinHistogram, BinsLocations],\
                               [EosinHistogram,       BinsLocations]]
    StainVectorsData        = [HematoxylinStainVectors,EosinStainVectors]    
    #-----------------------------------------------------------------
      
    Terminate(ResultsDirectory,HistogramData,StainVectorsData)
    print('****************************************')
    print('Done!')