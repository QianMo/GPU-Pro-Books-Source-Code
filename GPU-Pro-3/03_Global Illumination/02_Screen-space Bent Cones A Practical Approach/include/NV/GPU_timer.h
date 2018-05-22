// Simple GPU timer class that allows the timing of rendering without
// idling the hardware.
//
// Author: Evan Hart
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#ifndef GPU_TIMER_H
#define GPU_TIMER_H

#include <GL/glew.h>

//
//  Simple timer class using GL_EXT_timer_query. Used to track
// rendering time used on the GPU.
//
//////////////////////////////////////////////////////////////////////
class GPUtimer {
    GLuint _timers[1]; //allow 5 outstanding queries
    int _activeTimer;

public:
    
    // generate the queries
    void init() {
        glGenQueries( 1, _timers);
        _activeTimer = 0;
    }

    void cleanup() {
        glDeleteQueries( 1, _timers);
        _timers[0] = 0;
    }

    // begin the query, must not have nested timer queries
    void startTimer() {
        glBeginQuery( GL_TIME_ELAPSED_EXT, _timers[_activeTimer]);
    }

    // stop the query and advance to the next one as active
    void stopTimer() {
        glEndQuery( GL_TIME_ELAPSED_EXT);
        //_activeTimer = (_activeTimer + 1) % 5;
    }

    // return the most recent available timer result
    float getLatestTime() {
        int testNum = _activeTimer;
        GLint64EXT elapsedTime = 0;
        GLint queryReady = 0;

        do {
            //testNum = ( testNum + 4) % 5;
            glGetQueryObjectiv( _timers[testNum], GL_QUERY_RESULT_AVAILABLE, &queryReady);
        } while (!queryReady);

        glGetQueryObjecti64vEXT( _timers[testNum], GL_QUERY_RESULT, &elapsedTime);

        return elapsedTime / 1000000.0f; //convert to milliseconds
    }

};
#endif