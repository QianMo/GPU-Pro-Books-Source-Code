/************************************************************************/
/* frameRate.h                                                          */
/* ------------------                                                   */
/*                                                                      */
/* This is a pretty simple class that allows you to start timing at the */
/*    beginning of a frame's rendering and display the current          */
/*    framerate at the end of the frame.  Additionally, because 1-frame */
/*    counters are notoriously noisy (due to context switches, etc),    */
/*    this class allows the averaging over multiple frames, specified   */
/*    to the constructor.  Please note this class works only if the     */
/*    screen continuously updates (e.g., the idle function calls        */
/*    glutPostRedisplay()) -- after all a "framerate" makes no sense    */
/*    otherwise.                                                        */
/*                                                                      */
/* This code uses a "high resolution timer" I wrote to get sub-second   */
/*    timing values.  Note this high resolution timer is implementation */
/*    dependant, but there is code implementing it for Visual Studio,   */
/*    MacOS X, and Linux.  A standard "time()" function call could be   */
/*    substituted at the cost of less precision (and thus more noise in */
/*    the reported times.)                                              */
/*                                                                      */
/* Chris Wyman (12/7/2007)                                              */
/************************************************************************/


#ifndef __FRAMERATE_H__
#define __FRAMERATE_H__

#include "HighResolutionTimer.h"
#include <stdlib.h>

class FrameRate
{
private:
	int avgOverFrames;

	int currFrameCount;
	int frameIdx;
	float lastFrameRate;   // the last value computed in EndFrame().  Set to 0 initially.

	TimerStruct *timeArray;

public:
	// Initializes the frame rate counter.  The input will be the number
	//    of previous frames that framerate will be averaged over.
	FrameRate( int avgOverFrames = 5 );

	// Free allocated memory
	~FrameRate() { if (timeArray) free( timeArray ); }

	// Call at the very beginning of a frame
	void StartFrame( void );

	// Call at the end of a frame.  The value returned is the framerate
	//    (i.e., frames per second) averaged over the last N frames, where 
	//    N is the value specified in the constructor.  If the number of
	//    frames rendered is not yet N
	float EndFrame( void );

	// This returns the same value as the last call to EndFrame().  If you
	//    have not yet called EndFrame(), the return value is 0.
	inline float GetLastFrameRate( void )  { return lastFrameRate; }

};




#endif


