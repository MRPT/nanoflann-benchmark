#!/usr/bin/python3
import sys
import os
import matplotlib.pyplot as plt
import subprocess
import numpy as np

# calculate mean and standard deviation of an observation
def cal(MatrixTime, col):
    meanVal = 0.0
    stdVal  = 0.0
    numRepetitions = len(MatrixTime)
    for i in range(numRepetitions):
        meanVal = meanVal + MatrixTime[i][col][0]
    meanVal /= numRepetitions
    for i in range(numRepetitions):
        stdVal = stdVal + (MatrixTime[i][col][0]-meanVal)*(MatrixTime[i][col][0]-meanVal)
    stdVal /= numRepetitions
    stdVal = stdVal**0.5
    return meanVal, stdVal

# calculate build time and query time for a library
def plotTime(execPath, numRepetitions, numDivisions, numPoints):
    BuildTime = [[0.0 for x in range(numDivisions)] for y in range(numRepetitions)] 
    QueryTime = [[0.0 for x in range(numDivisions)] for y in range(numRepetitions)] 
    xaxis = []

    # run the process multiple times 
    for processCount in range(numRepetitions):
        proc = subprocess.Popen([execPath + ' ' + str(numPoints) + ' ' + str(processCount) ], stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        List = out.split()
        for it, item in enumerate(List):
            if(it<numDivisions):
                if(processCount==0):
                    xaxis += [int(item)]
            elif(it<2*numDivisions):
                BuildTime[processCount][it-numDivisions] = [float(item)*1e6]
            else:
                QueryTime[processCount][it-2*numDivisions] = [float(item)*1e6]
    BuildTimeFinal = []
    BuildTimeError = []
    QueryTimeFinal = []
    QueryTimeError = []
    for col in range(numDivisions):
        meanVal, stdVal = cal(BuildTime, col)
        BuildTimeFinal += [meanVal]
        BuildTimeError += [stdVal]
        meanVal, stdVal = cal(QueryTime, col)
        QueryTimeFinal += [meanVal]
        QueryTimeError += [stdVal]
    return xaxis, BuildTimeFinal, BuildTimeError, QueryTimeFinal, QueryTimeError

if __name__ == '__main__':

    if(len(sys.argv)!=6):
        raise ValueError('\n\n****Running Instructions:****\n ./randomTests.py numPoints nanoflannFlag flannFlag fastannFlag libkdtreeFlag\n\n Example:\n ./randomTests.py 1000 1 1 0 0\n This will run benchmarking tests with 1000 random points for nanoflann and flann.')

    numRepetitions = 50
    numDivisions   = 10
    # default states
    nanoflannFlag  = 0
    flannFlag      = 0
    fastannFlag    = 0
    libkdtreeFlag  = 0
    numPoints      = int(sys.argv[1])
    nanoflannFlag  = int(sys.argv[2])
    flannFlag      = int(sys.argv[3])
    fastannFlag    = int(sys.argv[4])
    libkdtreeFlag  = int(sys.argv[5])

    # BUILD TIME PLOTS

    fig, ax = plt.subplots()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path + '/../../build-Release/benchmarkTool/randomTests/'
    if(nanoflannFlag):
        xaxis, nanoflannBuildTimeFinal, nanoflannBuildTimeError, nanoflannQueryTimeFinal, nanoflannQueryTimeError = plotTime(dir_path + './benchmark_nanoflann_random', numRepetitions, numDivisions, numPoints)
        plt.plot(xaxis, nanoflannBuildTimeFinal, 'r', label='nanoflann', linewidth=3.0)
        plt.errorbar(xaxis, nanoflannBuildTimeFinal, color='k', yerr=nanoflannBuildTimeError, fmt='o', capsize=10)
    if(flannFlag):
        xOff = 5
        xaxis, flannBuildTimeFinal, flannBuildTimeError, flannQueryTimeFinal, flannQueryTimeError = plotTime(dir_path + './benchmark_flann_random', numRepetitions, numDivisions, numPoints)
        plt.plot(list(np.asarray(xaxis)+xOff), flannBuildTimeFinal, 'g', label='flann', linewidth=3.0)
        plt.errorbar(list(np.asarray(xaxis)+xOff), flannBuildTimeFinal, color='k', yerr=flannBuildTimeError, fmt='o', capsize=10)
    if(fastannFlag):
        xaxis, fastannBuildTimeFinal, fastannBuildTimeError, fastannQueryTimeFinal, fastannQueryTimeError = plotTime(dir_path + './benchmark_fastann_random', numRepetitions, numDivisions, numPoints)
        plt.plot(xaxis, fastannBuildTimeFinal, 'b', label='fastann', linewidth=3.0)
        plt.errorbar(xaxis, fastannBuildTimeFinal, color='k', yerr=fastannBuildTimeError, fmt='o', capsize=10)
    if(libkdtreeFlag):
        xaxis, libkdtreeBuildTimeFinal, libkdtreeBuildTimeError, libkdtreeQueryTimeFinal, libkdtreeQueryTimeError = plotTime(dir_path + './benchmark_libkdtree_random', numRepetitions, numDivisions, numPoints)
        plt.plot(xaxis, libkdtreeBuildTimeFinal, 'k', label='libkdtree', linewidth=3.0)
        plt.errorbar(xaxis, libkdtreeBuildTimeFinal, color='k', yerr=libkdtreeBuildTimeError, fmt='o', capsize=10)

    # plot configurations
    ax.grid(True)

    ticklines = ax.get_xticklines() + ax.get_yticklines()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

    for line in ticklines:
        line.set_linewidth(3)

    for line in gridlines:
        line.set_linestyle('-.')

    for label in ticklabels:
        label.set_color('k')
        label.set_fontsize('medium')

    plt.yscale("log") # vertical log scale
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel('Size of point cloud', fontsize=25)
    plt.ylabel('Time (μs)', fontsize=25)
    plt.title('kd-tree build time', fontsize=25)
    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper left', shadow=True)
    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')
    for label in legend.get_lines():
        label.set_linewidth(10)  # the legend line width
    plt.show()

    
    # QUERY TIME PLOTS

    fig, ax = plt.subplots()
    plt.yscale("linear")

    if(nanoflannFlag):
        plt.plot(xaxis, nanoflannQueryTimeFinal, 'r', label='nanoflann', linewidth=3.0)
        plt.errorbar(xaxis, nanoflannQueryTimeFinal, color='k', yerr=nanoflannQueryTimeError, fmt='o', capsize=10)
    if(flannFlag):
        xOff = 5
        plt.plot(list(np.asarray(xaxis)+xOff), flannQueryTimeFinal, 'g', label='flann', linewidth=3.0)
        plt.errorbar(list(np.asarray(xaxis)+xOff), flannQueryTimeFinal, color='k', yerr=flannQueryTimeError, fmt='o', capsize=10)
    if(fastannFlag):
        plt.plot(xaxis, fastannQueryTimeFinal, 'b', label='fastann', linewidth=3.0)
        plt.errorbar(xaxis, fastannQueryTimeFinal, color='k', yerr=fastannQueryTimeError, fmt='o', capsize=10)
    if(libkdtreeFlag):
        plt.plot(xaxis, libkdtreeQueryTimeFinal, 'k', label='libkdtree', linewidth=3.0)
        plt.errorbar(xaxis, libkdtreeQueryTimeFinal, color='k', yerr=libkdtreeQueryTimeError, fmt='o', capsize=10)

    # plot configurations
    ax.grid(True)

    ticklines = ax.get_xticklines() + ax.get_yticklines()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

    for line in ticklines:
        line.set_linewidth(3)

    for line in gridlines:
        line.set_linestyle('-.')

    for label in ticklabels:
        label.set_color('k')
        label.set_fontsize('medium')

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel('Size of point cloud', fontsize=25)
    plt.ylabel('Time (μs)', fontsize=25)
    plt.title('One 3d query time', fontsize=25)
    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper left', shadow=True)
    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')
    for label in legend.get_lines():
        label.set_linewidth(10)  # the legend line width
    plt.show()
