// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

// Helper class for time tracking of the app.

#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#ifdef WINDOWS
#include "windows.h"
#elif defined(LINUX)
#include <DataTypes.h>
#endif
#include <stdio.h>

class Performance
{
private:
	UINT						m_iFrameCount;											// Frame count for FPS
	LARGE_INTEGER				m_Timer;												// Timer for animation
	LARGE_INTEGER				m_FPSTimer;												// Timer for FPS
	LARGE_INTEGER				m_Timer_Frequency;										// Timer frequency (needed for tick to second convertion)

	LARGE_INTEGER				m_TimerRender;											// Timer for render
	LARGE_INTEGER				m_TimerFrequencyRender;									// Timer frequency (needed for tick to second convertion)

	__int64						m_iNumCandidates;
	__int64						m_iNumRays;												// Number of rays
public:
	// Constructor
	Performance(void);
	// Destructor
	~Performance(void);

	// Functions
	bool						updateFPS(char *aux, char *name);						// Update frames per second
	float						updateTime(void);										// Update current time
	float						updateTimeRender(void);									// Update render time (construct frame)
	void						addCandidates(__int64 n) { m_iNumCandidates += n; }
	void						addRays(__int64 n) { m_iNumRays += n; }

	// Getters
	UINT						getFrameCount() { return m_iFrameCount; }
	LARGE_INTEGER				getTimer() { return m_Timer; }
	LARGE_INTEGER				getFPSTimer() { return m_FPSTimer; }
	LARGE_INTEGER				getTimerFrequency() { return m_Timer_Frequency; }
	__int64						getNumCandidates() { return m_iNumCandidates; }
	__int64						getNumRays() { return m_iNumRays; }

	// Setters
	void						setNumCandidates(__int64 n) { m_iNumCandidates = n; }
	void						setNumRays(__int64 n) { m_iNumRays = n; }

};

#endif