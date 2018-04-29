/******************************************************************************

 @File         MDKTouch.h

 @Title        MDKTools

 @Copyright    Copyright (C) 2010 by Imagination Technologies Limited.

 @Platform     Independent

 @Description  TouchScreen interface and filtering functions
 
******************************************************************************/

#ifndef _MDK_TOUCH_H_
#define _MDK_TOUCH_H_

#if defined( __linux__ ) && !defined( X11BUILD ) && !defined( ANDROID ) && !defined( __PALMPDK__ )
#define EXPERIMENTAL_DYNAMIC_TOUCH
#endif


#if defined( PVRSHELL_OMAP3_TS_SUPPORT )

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>

#include "tslib.h"

#endif

#include "MDKPrecisionTimer.h"

class PVRShell;

namespace MDK {
	namespace Input {

		typedef float mdk_time;

		struct TouchSampleData
		{
			mdk_time time;
			float x, y;
			unsigned int pressure;

			TouchSampleData()
			{
				time = 0.0f;
				x = 0.0f;
				y = 0.0f;
				pressure = 0;
			}
			TouchSampleData(mdk_time time, float x, float y, unsigned int pressure)
			{
				this->time = time;
				this->x = x;
				this->y = y;
				this->pressure = pressure;
			}
		};


		class TouchDeque
		{
		private:
			TouchSampleData *data;
			unsigned int size;
			unsigned int front;
			unsigned int back;
			bool empty;

		public:
			explicit TouchDeque(const unsigned int capacity);
			~TouchDeque();

			void Add(mdk_time time, float x, float y, unsigned int pressure);

			void RemoveOlderThan(mdk_time time);

			unsigned int Prev(unsigned int i);
			unsigned int Next(unsigned int i);

			unsigned int Front() { return front; }
			unsigned int Back() { return back; }

			bool Empty();
			unsigned int Samples();

			TouchSampleData &Get(unsigned int i);
			TouchSampleData &GetFront();
			TouchSampleData &GetBack();

			void ToString(char *buff);
		};

		struct Rectangle
		{
			float left, bottom, right, top;

			Rectangle(float left, float bottom, float right, float top)
			{
				this->left = left;
				this->bottom = bottom;
				this->right = right;
				this->top = top;
			}
		};

		/*enum EFilterDevice {
			TS_FILTER_LAST_SAMPLE,
			TS_FILTER_AVERAGE,
			TS_FILTER_COUNT,
		};*/

		// High level state of the touchscreen device
		class TouchState
		{
			friend class TouchDevice;
		private:
			bool pressing;
			bool pressed;
			bool released;
			bool dragging;

			mdk_time startTime;
			mdk_time endTime;

			float startPosX, startPosY;
			float endPosX, endPosY;

			float motionX, motionY, motionAmp;
			float releaseMotionX, releaseMotionY;
			float inertiaX, inertiaY;

			float stdX, stdY, meanX, meanY;
			int motionSamples;

			// mdk_time-click related variables
			mdk_time start[2], end[2];
			int touchIndex;
			bool doubleTouch, singleTouch;
			bool gestureDrag;
			bool quit;

			bool SingleTouch(float fSingleTouchMaxOffset, float fSingleTouchMinDuration, float fSingleTouchMaxDuration) const;

		public:
			TouchState() {
				pressing = pressed = released = dragging = doubleTouch = singleTouch = gestureDrag = quit = false;

				startTime = endTime = 0.f;

				start[ 0 ] = start[ 1 ]  = end[ 0 ] = end[ 1 ] = 0.f;

				touchIndex = 0;

				startPosX = startPosY = endPosX = endPosY = motionX = motionY = motionAmp = releaseMotionX = releaseMotionY = 0.f;

				stdX = stdY = meanX = meanY = 0.f;

				motionSamples = 0;
			};

			float GetPositionX() const { return endPosX; };
			float GetPositionY() const { return endPosY; };

			float GetMotionX() const { return motionX; };
			float GetMotionY() const { return motionY; };

			float GetReleaseMotionX() const { return releaseMotionX; };
			float GetReleaseMotionY() const { return releaseMotionY; };

			float GetInertiaX() const { return inertiaX; }
			float GetInertiaY() const { return inertiaY; }


			bool IsPressing() const;
			bool IsPressed() const;
			bool IsReleased() const;
			bool IsDragging() const;

			bool SingleTouch() const { return singleTouch; }
			bool DoubleTouch() const { return doubleTouch; }
			bool GestureDrag() const { return gestureDrag; }

			mdk_time StartTime() const { return startTime; }
			mdk_time EndTime() const { return endTime; }

			bool InBox(const float &x0, const float &y0, const float &x1, const float &y1) const;
			bool InBox(const Rectangle &bounds) const;

			bool QuitEvent();
		};

		/*
		 * Main TouchDevice class. Provides initialization and input code, and functions to read the internal state.
		 */
		class TouchDevice
		{
		private:
			/* DEFINES */
			static const int TS_SAMPLES_TO_READ = 30;
			/* Values that can be changed on the config file */
			struct TSConfig
			{
				//! Size of the deque
				int iMaxSamplesCount;
				//! Time interval for removal from the queue
				float fInputTimeFrame;
				
				//! Time that needs to pass before input is considered as dragging.
				// This interval is also used to determined the "pressed" state
				float fPressedTimeDelay;
				
				//! Minimum time for single touch
				float fSingleTouchMinDuration;
				//! Max duration of single touch
				float fSingleTouchMaxDuration;
				//! Max offset for single touch
				float fSingleTouchMaxOffset;
				//! Minimum threshold that is applied to motion to cut off noise
				float fThresholdMultiplier;
				//! Inertia of the system
				float fInertiaTau;
				//! When calculating the motion, apply some attenuation to the first samples
				float fMotionAttenuationTau;
				//! Array defining the bounds of the rectangle of the quit area
				float fQuitArea[4];
			} tsConfig;

			void ReadConfig(char *szConfigFile);

	#if defined( PVRSHELL_OMAP3_TS_SUPPORT ) || defined( EXPERIMENTAL_DYNAMIC_TOUCH )
			struct ts_sample *samples;
			struct tsdev *ts;

			bool initTimer;
			unsigned int iStartSec;
			unsigned int iStartUSec;
			bool bReadRaw;
			// fTimeDiff stores the difference between the shell time and the internal touchscreen time
			mdk_time fTimeDiff;
	#endif
			//! Timer used internally by various functions
			PrecisionTimer timer;

			//! Configuration file to read from and write to
			static const char c_szConfigFile[50];

			//! Horizontal resolution of the viewport (used for normalizing coordinates)
			float fWidth;
			//! Vertical resolution of the viewport (used for normalizing coordinates)
			float fHeight;
			//! Tells whether the viewport is rotated or not
			bool bRotated;
	
			//! Pointer that will store the queue of samples
			TouchDeque *dqSamples;

			//! Internal touchscreen state
			TouchState state;

			//! Pointer for accessing shell touchscreen events
			PVRShell* m_pShell;

			//! Update the TouchState struct. Called inside Input(); calls UpdateMeanAndVar
			bool UpdateState(mdk_time curTime, bool pressed);
			//! Updates mean and standard deviation used for motion estimation
			void UpdateMeanAndVar(mdk_time curTime, bool pressed);
			//! Updates mean and standard deviation used for motion estimation 
			void MeanAndVar(unsigned int numSamples, float &meanX, float &meanY, float &stdX, float &stdY);

			//! Temporary structure for storing motions between two samples
			struct Motion
			{
				float dx;
				float dy;
			} *motion;

			float guessedMotionX, guessedMotionY;

			// Internal function to convert from integer to float coordinates in the range [-1, 1]
			void ToScreenXY(float tx, float ty, float &x, float &y);


			TouchDevice();
			~TouchDevice();
		public:
			static TouchDevice &Instance();

			//! Gets internal state
			const TouchState& GetState() const { return state; };

			//! Initialization
			bool Init(PVRShell* pShell, int width, int height, bool rotated = false);
			//! Sets the rotation of the viewport
			void SetRotated(bool rotated) { bRotated = rotated; }
			//! Synchronization of the application timer with the touchscreen internal timer
			// The current implementation just starts the timer
			bool Synchronize(Timer *pTimer = NULL);

			//! Reads samples in the current frame and updates TouchState
			// Returns false if a quit event happened
			bool Input();

			// DEPRECATED functions
			// Public conversion of the touch position into viewport coordinates
			//void ToScreen(float &sX, float &sY, EFilterDevice filter = TS_FILTER_LAST_SAMPLE);
			//bool Predict(float &predX, float &predY, mdk_time curTime, mdk_time deltaTime);

			//! Returns number of samples currently in the queue
			int Samples() { return dqSamples->Samples(); }

			// Note: It has to be x1 >= x0 and y1 >= y0 to have a valid box region
			bool InBox(const float &x0, const float &y0, const float &x1, const float &y1) const;
			bool InBox(const Rectangle &bounds) const;

			/* Functions for access to internal state */
			bool SingleTouch() const { return state.SingleTouch(); }
			bool DoubleTouch() const { return state.DoubleTouch(); }
			bool GestureDrag() const;

			/* Get Functions */
			float GetPositionX() const { return state.endPosX; }
			float GetPositionY() const { return state.endPosY; }

			float GetMeanX() const { return state.meanX; }
			float GetMeanY() const { return state.meanY; }

			float GetMotionX() const { return state.motionX; }
			float GetMotionY() const { return state.motionY; }

			float GetReleaseMotionX() const { return state.releaseMotionX; }
			float GetReleaseMotionY() const { return state.releaseMotionY; }

			float GetInertiaX() const { return state.inertiaX; }
			float GetInertiaY() const { return state.inertiaY; }

			float GetThresholdX() const;
			float GetThresholdY() const;

			mdk_time StartTime() const { return state.StartTime(); }
			mdk_time EndTime() const { return state.EndTime(); }

			bool IsPressing() const { return state.IsPressing(); }
			bool IsPressed() const { return state.IsPressed(); }
			bool IsReleased() const { return state.IsReleased(); }
			bool IsDragging()  const { return state.IsDragging(); }

			bool QuitEvent() { return state.QuitEvent(); }

			//! Print out touchscreen internal state
			void ToString(char *buffer) { dqSamples->ToString(buffer); }			

			//! Utility function for calibrating the touchscreen device
			bool Calibrate();
		};
	}
}
#endif

