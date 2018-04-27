/**
 *
 *  This software module was originally developed for research purposes,
 *  by Multimedia Lab at Ghent University (Belgium).
 *  Its performance may not be optimized for specific applications.
 *
 *  Those intending to use this software module in hardware or software products
 *  are advized that its use may infringe existing patents. The developers of 
 *  this software module, their companies, Ghent Universtity, nor Multimedia Lab 
 *  have any liability for use of this software module or modifications thereof.
 *
 *  Ghent University and Multimedia Lab (Belgium) retain full right to modify and
 *  use the code for their own purpose, assign or donate the code to a third
 *  party, and to inhibit third parties from using the code for their products. 
 *
 *  This copyright notice must be included in all copies or derivative works.
 *
 *  For information on its use, applications and associated permission for use,
 *  please contact Prof. Rik Van de Walle (rik.vandewalle@ugent.be). 
 *
 *  Detailed information on the activities of
 *  Ghent University Multimedia Lab can be found at
 *  http://multimedialab.elis.ugent.be/.
 *
 *  Copyright (c) Ghent University 2004-2009.
 *
 **/

#pragma once

class klStatisticsCounter {
protected:
    double *history;
    int historySize;
    int offset;
    char theString[256];
    FILE *logFile;
    LARGE_INTEGER counterFrequency;
public:

    /*
        Create a new statistics counter, historySize determines the number of samples
        in the history (used for min,max,avg).
        If logName is nonNull the statistics can easily be saved to a log file using the
        "log" function.
    */
    klStatisticsCounter(int historySize = 100, char *logName = NULL) {
        QueryPerformanceFrequency(&counterFrequency);
        this->historySize = historySize;
        history = (double *)malloc(sizeof(double)*historySize);
        memset(history,0,sizeof(double)*historySize);
        offset = 0;
        if ( logName != NULL ) {
            logFile = fopen(logName,"w");
        } else {
            logFile = NULL;
        }
    }

    ~klStatisticsCounter(void) {
        if ( logFile ) fclose(logFile);
        free(history);
    }

    void log(void) {
        if (logFile) {
            fputs(getString(),logFile);
            fputs("\n",logFile);
        }
    }

    void put(double d) {
        offset = (offset+1)%historySize;
        history[offset] = d;
    }

    void putCounts(long long time) {
        double inSeconds = (double)(time)/(double)(counterFrequency.QuadPart);
        put(inSeconds);
    }

    float getAvg(void) {
        double total = 0;
        for ( int i=0; i<historySize; i++ ) {
            total += history[i];
        }
        return (float)(total / (double)historySize);
    }

    float getMin(void) {
        double minf = history[0];
        for ( int i=1; i<historySize; i++ ) {
            minf = (minf < history[i]) ? minf : history[i];
        }
        return (float)minf;
    }

    float getMax(void) {
        double maxf = history[0];
        for ( int i=1; i<historySize; i++ ) {
            maxf = (maxf > history[i]) ? maxf : history[i];
        }
        return (float)maxf;
    }

    float getLast(void) {
        return (float)history[offset];
    }

    /*
        Returns a handy string with all the info, the string is owned by the counter so don't worry
        about freeing it.
    */
    char *getString(void) {
        sprintf(theString,"last:%.2f min:%.2f max:%.2f avg:%.2f",getLast(),getMin(),getMax(),getAvg());
        return theString;
    }
};

class klFpsCounter : public klStatisticsCounter {
    LARGE_INTEGER lastTime;
public:

    klFpsCounter(int _historySize = 100, char *_logName = NULL) : klStatisticsCounter(_historySize,_logName) {
        QueryPerformanceCounter(&lastTime);
    }

    void frame(void) {
        LARGE_INTEGER time;
        QueryPerformanceCounter(&time);
        double deltaInSeconds = (double)(time.QuadPart - lastTime.QuadPart)/(double)(counterFrequency.QuadPart);
        lastTime = time;
        //double fps = 1.0f / deltaInSeconds;
        put(deltaInSeconds*1000.0);
    }
};
