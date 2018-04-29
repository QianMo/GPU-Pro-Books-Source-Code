/******************************************************************************

 @File         MDKTouch.cpp

 @Title        MDKTools

 @Copyright    Copyright (C) 2010 by Imagination Technologies Limited.

 @Platform     Independent

 @Description  TouchScreen interface and filtering functions
 
******************************************************************************/

#include "../PVRTResourceFile.h"

#include "MDKTouch.h"
#include "MDKMisc.h"
#include <math.h>
#include "Resource.h"

#include "PVRShell.h"

//#define PRINT_DIAGNOSTIC_INFORMATION

namespace MDK {
	namespace Input {

	#if defined( EXPERIMENTAL_DYNAMIC_TOUCH )
		#include <stdarg.h>
		#include <sys/time.h>
		#include <dlfcn.h>

		struct tsdev;

		struct ts_sample {
			int		x;
			int		y;
			unsigned int	pressure;
			struct timeval	tv;
		};

		extern int (*ts_error_fn)(const char *fmt, va_list ap);

		int (*ts_close)(struct tsdev *);
		int (*ts_config)(struct tsdev *);
		int (*ts_fd)(struct tsdev *);
		int (*ts_load_module)(struct tsdev *, const char *mod, const char *params);
		struct tsdev *(*ts_open)(const char *dev_name, int nonblock);
		int (*ts_read)(struct tsdev *, struct ts_sample *, int);
		int (*ts_read_raw)(struct tsdev *, struct ts_sample *, int);	

		bool TryTSLib()
		{
			DebugPrint( "Opening libts.so...\n" );
			void *pLib = dlopen( "libts.so", RTLD_LAZY | RTLD_GLOBAL );

			if( pLib == 0 )
			{
				DebugPrint( "libts.so not found." );
				return false;
			}

			DebugPrint("Opened.\n" );

			DebugPrint( "Getting symbol ts_config.\n" );
			ts_config = (int (*)(tsdev*))dlsym( pLib, "ts_config" );
			DebugPrint( "Getting symbol ts_read.\n" );
			ts_read = (int (*)(tsdev*, ts_sample*, int))dlsym( pLib, "ts_read" );
			DebugPrint( "Getting symbol ts_read_raw.\n" );
			ts_read_raw = (int (*)(tsdev*, ts_sample*, int))dlsym( pLib, "ts_read_raw" );
			DebugPrint( "Getting symbol ts_open.\n" );
			ts_open = (struct tsdev *(*)(const char *, int))dlsym( pLib, "ts_open" );

			if( ts_config == 0 || ts_read == 0 || ts_read_raw == 0 )
			{
				DebugPrint( "Failed to get symbols.\n" );
				dlclose( pLib );
				return false;
			}

			DebugPrint( "tslib symbols acquired.\n" );

			return true;
		}
	#endif

		const char TouchDevice::c_szConfigFile[] = "data/ts_config.txt";

		enum ETSPlatforms { MDK_TS_OMAP_ZOOM1, MDK_TS_OMAP_ZOOM2, MDK_TS_DEFAULT, MDT_TS_PLATFORM_COUNT };

	#ifdef PVRSHELL_OMAP3_TS_SUPPORT
		#if defined DEVICE_ZOOM1
			static ETSPlatforms thisPlatform = MDK_TS_OMAP_ZOOM1;
		#elif defined DEVICE_ZOOM2
			static ETSPlatforms thisPlatform = MDK_TS_OMAP_ZOOM2;
		#else
			static ETSPlatforms thisPlatform = MDK_TS_DEFAULT;
		#endif
	#else
		static ETSPlatforms thisPlatform = MDK_TS_DEFAULT;
	#endif

		//! Some specific platforms are affected by noise and this affects motion estimation.
		// A platform specific threshold can be set to ensure the motion is set to zero when necessary.
		// The threshold is equal to 3 times the standard deviation observed on a set of samples
		// read from the platform while the input position was fixed.
		float afPlatform3Sigma[MDT_TS_PLATFORM_COUNT][2] = {
			{ 0.0130f, 0.0065f },
			{ 0.0036f, 0.0060f },
			{ 0.0000f, 0.0000f }
		};

		/***********************************************************************
		 * TouchDeque Implementation
		 ***********************************************************************/
		TouchDeque::TouchDeque(const unsigned int capacity)
		{
			size = capacity;
			data = new TouchSampleData[size];

			front = size - 1;
			back = 0;
			empty = true;
		}
		TouchDeque::~TouchDeque()
		{
			SAFE_DELETE_ARRAY(data)
		}

		void TouchDeque::Add(mdk_time time, float x, float y, unsigned int pressure)
		{
			// check whether the queue is full. If so, move the back
			if ( ( empty == false ) && ( front == back - 1 || ( front == size - 1 && back == 0 ) ) )
			{
				if (++back == size)
					back = 0;
			}
			if (empty)
				empty = false;

			if (++front == size)
				front = 0;

			data[front] = TouchSampleData(time, x, y, pressure);
		}

		void TouchDeque::RemoveOlderThan(mdk_time time)
		{
			if (empty)
				return;

			bool exitCondition = false; 
			while (data[back].time < time)
			{
				exitCondition = back == front;
				if (++back == size)
					back = 0;
				
				if (exitCondition)
					break;			
			}
			if (exitCondition)
				empty = true;
		}
			
		unsigned int TouchDeque::Prev(unsigned int i)
		{
			return i == 0 ? size - 1 : i - 1;
		}
		unsigned int TouchDeque::Next(unsigned int i)
		{
			return i == size - 1 ? 0 : i + 1;
		}

		bool TouchDeque::Empty()
		{
			return empty;
		}

		unsigned int TouchDeque::Samples()
		{
			if (empty)
				return 0;

			if (front < back)
				return front + 1 + size - back;

			return front - back + 1;
		}

		TouchSampleData &TouchDeque::Get(unsigned int i)
		{
			return data[i];
		}

		TouchSampleData &TouchDeque::GetFront()
		{
			return data[Front()];
		}

		TouchSampleData &TouchDeque::GetBack()
		{
			return data[Back()];
		}

		void TouchDeque::ToString(char *buff)
		{
			sprintf(buff, "EMPTY=%d\nBACK=%d TIME=%.3f\nFRONT=%d TIME=%.3f", Empty(), Back(), data[Back()].time, Front(), data[Front()].time);
		}


		/***********************************************************************
		 * TouchDevice Implementation
		 ***********************************************************************/

		void TouchDevice::ReadConfig(char *szConfigFile)
		{
			// Samples buffer size
			tsConfig.iMaxSamplesCount = 500;
			// Input interval where the incoming samples are used
			tsConfig.fInputTimeFrame = 0.15f;
			// High level control parameters
			tsConfig.fPressedTimeDelay = 0.05f;
			tsConfig.fSingleTouchMinDuration = 0.03f;

			tsConfig.fSingleTouchMaxOffset = 0.1f;
			tsConfig.fSingleTouchMaxDuration = 0.15f;
			tsConfig.fThresholdMultiplier = 1.0f;
			tsConfig.fMotionAttenuationTau = 15.0f;
			tsConfig.fInertiaTau = 2.0f;

			tsConfig.fQuitArea[0] = 0.75f;
			tsConfig.fQuitArea[1] = 0.75f;
			tsConfig.fQuitArea[2] = 1.0f;
			tsConfig.fQuitArea[3] = 1.0f;

			char sBuf[200];
			FILE *fp = fopen(szConfigFile, "r");
			if (fp == NULL)
			{
				fp = fopen(szConfigFile, "wb");
				if (fp!=0)
				{
					sprintf(sBuf, "MAX_SAMPLES_COUNT      %d\n", tsConfig.iMaxSamplesCount);
					fwrite(sBuf, 1 , strlen(sBuf) , fp);
					sprintf(sBuf, "INPUT_TIME_FRAME       %.2f\n", tsConfig.fInputTimeFrame);
					fwrite(sBuf, 1 , strlen(sBuf) , fp);
					sprintf(sBuf, "PRESSED_TIME_DELAY     %.2f\n", tsConfig.fPressedTimeDelay);
					fwrite(sBuf, 1 , strlen(sBuf) , fp);
					sprintf(sBuf, "TOUCH_MIN_DURATION     %.2f\n", tsConfig.fSingleTouchMinDuration);
					fwrite(sBuf, 1 , strlen(sBuf) , fp);
					sprintf(sBuf, "TOUCH_MAX_DURATION     %.2f\n", tsConfig.fSingleTouchMaxDuration);
					fwrite(sBuf, 1 , strlen(sBuf) , fp);
					sprintf(sBuf, "TOUCH_MAX_OFFSET       %.2f\n", tsConfig.fSingleTouchMaxOffset);
					fwrite(sBuf, 1 , strlen(sBuf) , fp);
					sprintf(sBuf, "THRESHOLD_MULTIPLIER   %.2f\n", tsConfig.fThresholdMultiplier);
					fwrite(sBuf, 1 , strlen(sBuf) , fp);
					sprintf(sBuf, "INERTIA_TAU            %.2f\n", tsConfig.fInertiaTau);
					fwrite(sBuf, 1 , strlen(sBuf) , fp);
					sprintf(sBuf, "MOTION_ATTENUATION_TAU %.2f\n", tsConfig.fMotionAttenuationTau);
					fwrite(sBuf, 1 , strlen(sBuf) , fp);
					sprintf(sBuf, "QUIT_AREA              %.2f %.2f %.2f %.2f\n", tsConfig.fQuitArea[0], tsConfig.fQuitArea[1], tsConfig.fQuitArea[2], tsConfig.fQuitArea[3]);
					fwrite(sBuf, 1 , strlen(sBuf) , fp);
				
					fclose(fp);
				}
			}
			else
			{
				while(fscanf(fp, "%s", sBuf) != EOF)
				{
					if (!strcmp(sBuf, "MAX_SAMPLES_COUNT"))
						fscanf(fp, "%d", &tsConfig.iMaxSamplesCount);
					else if (!strcmp(sBuf, "INPUT_TIME_FRAME"))
						fscanf(fp, "%f", &tsConfig.fInputTimeFrame);
					else if (!strcmp(sBuf, "PRESSED_TIME_DELAY"))
						fscanf(fp, "%f", &tsConfig.fPressedTimeDelay);
					else if (!strcmp(sBuf, "TOUCH_TIME_DELAY"))
						fscanf(fp, "%f", &tsConfig.fSingleTouchMinDuration);
					else if (!strcmp(sBuf, "TOUCH_MAX_OFFSET"))
						fscanf(fp, "%f", &tsConfig.fSingleTouchMaxOffset);
					else if (!strcmp(sBuf, "TOUCH_MAX_TIME"))
						fscanf(fp, "%f", &tsConfig.fSingleTouchMaxDuration);
					else if (!strcmp(sBuf, "THRESHOLD_MULTIPLIER"))
						fscanf(fp, "%f", &tsConfig.fThresholdMultiplier);
					else if (!strcmp(sBuf, "INERTIA_TAU"))
						fscanf(fp, "%f", &tsConfig.fInertiaTau);
					else if (!strcmp(sBuf, "MOTION_ATTENUATION_TAU"))
						fscanf(fp, "%f", &tsConfig.fMotionAttenuationTau);
					else if (!strcmp(sBuf, "QUIT_AREA"))
						fscanf(fp, "%f %f %f %f", &tsConfig.fQuitArea[0], &tsConfig.fQuitArea[1], &tsConfig.fQuitArea[2], &tsConfig.fQuitArea[3]);
				}
				fclose(fp);
			}
		}

		/********************************************************************
		 * TouchDevice Implementation
		 ********************************************************************/
		TouchDevice &TouchDevice::Instance()
		{
			static TouchDevice device;
			return device;
		}
		TouchDevice::TouchDevice() {

			dqSamples = NULL;
			motion = NULL;
		#if defined( PVRSHELL_OMAP3_TS_SUPPORT ) || defined( EXPERIMENTAL_DYNAMIC_TOUCH )
			initTimer = false;
			bReadRaw = false;
			samples = NULL;
		#else
			// No mouse initialisation needs to be added

		#endif
			//state.m_pDevice=this;
		}

		TouchDevice::~TouchDevice()
		{
		#if defined( PVRSHELL_OMAP3_TS_SUPPORT ) || defined( EXPERIMENTAL_DYNAMIC_TOUCH )
			SAFE_DELETE_ARRAY( samples );
		#endif

			SAFE_DELETE_ARRAY( motion );
			SAFE_DELETE( dqSamples );
		}

		bool TouchDevice::Init(PVRShell* pShell, int width, int height, bool rotated/* = false*/)
		{
			fWidth = (float)width;
			fHeight = (float)height;
			bRotated = rotated;
			
			m_pShell=pShell;

			char sBuf[200];
			if (CPVRTResourceFile::GetReadPath().empty())
				sprintf(sBuf, "%s/%s", (char*)m_pShell->PVRShellGet(prefReadPath), c_szConfigFile);
			else
				sprintf(sBuf, "%s/%s", CPVRTResourceFile::GetReadPath().c_str(), c_szConfigFile);
			ReadConfig(sBuf);

			dqSamples = new TouchDeque(tsConfig.iMaxSamplesCount);
			motion = new Motion[tsConfig.iMaxSamplesCount];

		#if defined( PVRSHELL_OMAP3_TS_SUPPORT )
			samples = new ts_sample[TS_SAMPLES_TO_READ];
			bReadRaw = false;

			/*************************************************
			 * NOTE: For the init code to work, these variables have to be set prior to the app launch.
			 *
			 * export TSLIB_TSDEVICE=/dev/input/event1
			 * export TSLIB_CONFFILE=/etc/ts.conf
			 * export TSLIB_CALIBFILE=/etc/pointercal
			 * export TSLIB_CONSOLEDEVICE=/dev/tty
			 * export TSLIB_FBDEVICE=/dev/fb0
			 *************************************************/

			ts = ts_open("/dev/input/event1", 1);

			if( !ts )
			{
				DebugPrint("ts_open failed.\n");
				return false;
			}

			if( ts_config( ts ) )
			{
				DebugPrint("ts_config failed.\n");
				return false;
			}

			DebugPrint("TouchScreen Initialized.\n");

		#elif defined( EXPERIMENTAL_DYNAMIC_TOUCH )
			if( TryTSLib() == false )
			{
				return false;
			}

			samples = new ts_sample[TS_SAMPLES_TO_READ];
			bReadRaw = false;

			ts = ts_open("/dev/input/event1", 1);

			if( !ts )
			{
				DebugPrint("ts_open failed.\n");
				return false;
			}

			if( ts_config( ts ) )
			{
				DebugPrint("ts_config failed.\n");
				return false;
			}

			DebugPrint("TouchScreen Initialized.\n");


			thisPlatform = MDK_TS_DEFAULT;

			const char *pPlatform = GetEnv( "MDK_DEVICE_HINT" );

			if( pPlatform != 0 )
			{
				if( strcmp( pPlatform, "ZOOM1" ) == 0 )
				{
					thisPlatform = MDK_TS_OMAP_ZOOM1;
				} 
				else if( strcmp( pPlatform, "ZOOM2" ) == 0 )
				{
					thisPlatform = MDK_TS_OMAP_ZOOM2;
				}
			}

		#else
			// The mouse subsystem is automatically initialized by the OS so nothing has to be done here.
	
		#endif

			return true;
		}

		bool TouchDevice::Synchronize(Timer *pTimer/* = NULL*/)
		{	
		#if defined( PVRSHELL_OMAP3_TS_SUPPORT ) || defined( EXPERIMENTAL_DYNAMIC_TOUCH )
			// Remove old samples from the internal buffer
			int ret, oldSamples = 0;
			do
			{
				ret = bReadRaw ? ts_read_raw(ts, samples, TS_SAMPLES_TO_READ) : ts_read(ts, samples, TS_SAMPLES_TO_READ);
				oldSamples += ret;
			}
			while (ret > 0);
		#endif
			timer.Start();

			if (pTimer)
				pTimer->Restart();

			return true;
		}

		bool TouchDevice::Input()
		{
			timer.Update();
			float curTime = timer.GetTimef();
		#if defined( PVRSHELL_OMAP3_TS_SUPPORT ) || defined( EXPERIMENTAL_DYNAMIC_TOUCH )
			int ret = bReadRaw ? ts_read_raw(ts, samples, TS_SAMPLES_TO_READ) : ts_read(ts, samples, TS_SAMPLES_TO_READ);
	
			if (ret < 0 && !bReadRaw)
				return false;	

		#ifdef PRINT_DIAGNOSTIC_INFORMATION
			static int iFrameCounter = 0;
			iFrameCounter++;
		#endif

			// Get Sample
			if (ret >= 1)
			{
				if (initTimer == false)
				{
					iStartSec = samples[0].tv.tv_sec;
					iStartUSec = samples[0].tv.tv_usec;
					fTimeDiff = timer.GetTimef();
					initTimer = true;
				}
		#ifdef PRINT_DIAGNOSTIC_INFORMATION
				DebugPrint("Frame %d, samples=%d\n", iFrameCounter, ret);
		#endif

				// Copy all the samples in the circular buffer
				for (int i = 0; i < ret; i++)
				{
					struct ts_sample &samp = samples[i];

		#ifdef PRINT_DIAGNOSTIC_INFORMATION
					DebugPrint("%ld.%06ld: %6d %6d %6d\n", samp.tv.tv_sec - iStartSec, samp.tv.tv_usec - iStartUSec, samp.x, samp.y, samp.pressure);
		#endif
					float x, y;
					ToScreenXY(samp.x, samp.y, x, y);
					mdk_time time = PrecisionTimer::ToSeconds(samp.tv.tv_sec - iStartSec, samp.tv.tv_usec - iStartUSec) + fTimeDiff;
					dqSamples->Add(time, x, y, samp.pressure);
				}
			}
			dqSamples->RemoveOlderThan(curTime - tsConfig.fInputTimeFrame);

			// if ret >= 1, there were samples in the internal buffer
			return UpdateState(curTime, ret >= 1);

		#else

			bool pressed = m_pShell->PVRShellGet(prefButtonState) ? true : false;
			float *fTouch = (float*)m_pShell->PVRShellGet(prefPointerLocation); 

			if (pressed && fTouch)
			{
				float tx = fTouch[ 0 ];
				float ty = fTouch[ 1 ];

#if defined( __BADA__ ) || defined( __PALMPDK__ )
				if( bRotated )
				{
					Swap( tx, ty );
					ty = 1.f - ty;
				}
#endif

				float x, y;
				ToScreenXY( tx, ty, x, y );
				dqSamples->Add( curTime, x, y, 0 );
			}
			dqSamples->RemoveOlderThan(curTime - tsConfig.fInputTimeFrame);

			return UpdateState(curTime, pressed && fTouch != NULL);
		#endif
		}

		bool TouchDevice::UpdateState(mdk_time curTime, bool pressed)
		{
			TouchSampleData &lastSample = dqSamples->Get(dqSamples->Front());

			//Update last sample
			if (pressed)
			{
				state.endTime = lastSample.time;
				state.endPosX = lastSample.x;
				state.endPosY = lastSample.y;
			}

			UpdateMeanAndVar(curTime, pressed);

			state.released = false;
			state.pressed = false;
			state.doubleTouch = false;

			if (pressed)
			{
				state.motionX = state.meanX;
				state.motionY = state.meanY;

				// Apply threshold
				if (fabs(state.motionX) < afPlatform3Sigma[thisPlatform][0] * tsConfig.fThresholdMultiplier)
					state.motionX = 0.0f;
				if (fabs(state.motionY) < afPlatform3Sigma[thisPlatform][1] * tsConfig.fThresholdMultiplier)
					state.motionY = 0.0f;

				state.motionAmp = (float)sqrt(state.motionX * state.motionX + state.motionY * state.motionY);
				// Start new acquisition
				if (!state.pressing)
				{
					//brt_printf("noTouchFrames = %d\n", noTouchFrames);

					state.pressing = true;
					state.pressed = true;
					state.released = false;
					state.dragging = false;

					state.startTime = lastSample.time;
					state.startPosX = lastSample.x;
					state.startPosY = lastSample.y;

					// Double click
					if (state.touchIndex == 1)
					{
						if (state.startTime - state.end[0] > 0.2)
						{
							state.touchIndex = 0;
						}
					}
					state.start[state.touchIndex] = state.startTime;

					// state.motionX and state.motionY should be 0
				}
				else
				{
					// after tsConfig.fPressedTimeDelay, dragging is considered on
					if (state.endTime > state.startTime + tsConfig.fPressedTimeDelay)
					{
						state.dragging = true;
					}
				}
			}
			// !pressed, old state == pressing
			else if (state.pressing)
			{
				if (state.endTime < curTime - tsConfig.fPressedTimeDelay)
				{
					//brt_printf("noTouchFrames = %d\n", noTouchFrames);

					state.pressing = false;
					state.pressed = false;
					state.released = true;
					state.dragging = false;

					state.releaseMotionX = state.motionX;
					state.releaseMotionY = state.motionY;

					state.motionX = 0.0f;
					state.motionY = 0.0f;
					state.motionAmp = 0.0f;

					// Double click
					state.end[state.touchIndex] = state.endTime;
					if (state.end[state.touchIndex] - state.start[state.touchIndex] < 0.2)
					{
						if (state.touchIndex == 0)
						{
							state.touchIndex = 1;
						}
						else
						{
							state.doubleTouch = true;
							// lower right corner
							state.quit = InBox(tsConfig.fQuitArea[0], tsConfig.fQuitArea[1], tsConfig.fQuitArea[2], tsConfig.fQuitArea[3]);
							state.touchIndex = 0;
						}
					}
					else
					{
						state.touchIndex = 0;
					}
				}
			}
		#ifdef PRINT_DIAGNOSTIC_INFORMATION
			//if (state.endTime - (curTime - tsConfig.fPressedTimeDelay) > -tsConfig.fPressedTimeDelay)
			//	brt_printf("Delay=%.3f, EndTime=%.3f, CurTime=%.3f, Diff=%.3f\n", tsConfig.fPressedTimeDelay, state.endTime, curTime, state.endTime - (curTime - tsConfig.fPressedTimeDelay));
		#endif
			state.singleTouch = state.SingleTouch( tsConfig.fSingleTouchMaxOffset, tsConfig.fSingleTouchMinDuration, tsConfig.fSingleTouchMaxDuration );
			state.gestureDrag = GestureDrag();

			float inertia = Inertia(curTime - state.endTime, tsConfig.fInertiaTau);
			// This branch takes into account discontinuities in the signal when actual TS devices are used.
			if (state.pressing && !pressed)
			{
				state.inertiaX = state.motionX * inertia;
				state.inertiaY = state.motionY * inertia;
			}
			else
			{
				state.inertiaX = state.releaseMotionX * inertia;
				state.inertiaY = state.releaseMotionY * inertia;
			}

			return true;
		}

		void TouchDevice::MeanAndVar(unsigned int numSamples, float &meanX, float &meanY, float &stdX, float &stdY)
		{
			meanX = meanY = stdX = stdY = 0.0f;
			unsigned int i;
			// Mean calculation
			for (i = 0; i < numSamples; i++)
			{
				meanX += motion[i].dx;
				meanY += motion[i].dy;
			}
			meanX /= (float)numSamples;
			meanY /= (float)numSamples;

			float x, y;
			for (i = 0; i < numSamples; i++)
			{
				x = motion[i].dx - meanX;
				y = motion[i].dy - meanY;
				stdX += x * x;
				stdY += y * y;
			}
			stdX /= (float)numSamples;
			stdY /= (float)numSamples;
			stdX = (float)sqrt(stdX);
			stdY = (float)sqrt(stdY);
		}

		void TouchDevice::UpdateMeanAndVar(mdk_time curTime, bool pressed)
		{
			// Initialize to 0 regardless of any condition
			state.motionSamples = 0;
			state.meanX = state.meanY = state.stdX = state.stdY = 0.0f;
		
			if (dqSamples->Samples() < 2)
			{
				return;
			}

			// If this point is reached there is at least one sample as firstSample != lastSample

			mdk_time prevT, dt;

			unsigned int prevSampleIndex, curSampleIndex = dqSamples->Front();

			// Copy all the samples in an array
			do
			{
				prevSampleIndex = dqSamples->Prev(curSampleIndex);

				TouchSampleData &curSample = dqSamples->Get(curSampleIndex);
				TouchSampleData &prevSample = dqSamples->Get(prevSampleIndex);
				prevT = prevSample.time;

				if (curSample.time > prevSample.time)
				{
					dt = curSample.time - prevSample.time;

					// attenuate the motion if it has just started
					float attenuation = Exp(curSample.time - state.startTime, 1.0f, tsConfig.fMotionAttenuationTau) *
										Exp(curTime - curSample.time, 1.0f, tsConfig.fMotionAttenuationTau);

					motion[state.motionSamples].dx = attenuation * (curSample.x - prevSample.x) / dt;
					motion[state.motionSamples].dy = attenuation * (curSample.y - prevSample.y) / dt;
				
					state.motionSamples++;
				}
				curSampleIndex = prevSampleIndex;
			}
			while (prevSampleIndex != dqSamples->Back() && prevT >= curTime - tsConfig.fInputTimeFrame);

			if (state.motionSamples <= 1)
			{
				return;
			}
			MeanAndVar(state.motionSamples, state.meanX, state.meanY, state.stdX, state.stdY);
		}

		void TouchDevice::ToScreenXY(float tx, float ty, float &x, float &y)
		{
		#if defined( PVRSHELL_OMAP3_TS_SUPPORT ) || defined( EXPERIMENTAL_DYNAMIC_TOUCH )
			if (bRotated)
			{
				y = 1.0f - (tx / fWidth) * 2.0f;
				x = 1.0f - (ty / fHeight) * 2.0f;
			}
			else
			{
				x = (tx / fWidth) * 2.0f - 1.0f;
				y = 1.0f - (ty / fHeight) * 2.0f;
			}
		#else
			if (bRotated)
			{
				x = 1.0f - ty * 2.0f;
				y = 1.0f - tx * 2.0f;
			}
			else
			{
				x = tx * 2.0f - 1.0f;
				y = 1.0f - ty * 2.0f;
			}
		#endif
		}

		float TouchDevice::GetThresholdX() const
		{
			return afPlatform3Sigma[thisPlatform][0] * tsConfig.fThresholdMultiplier;
		}

		float TouchDevice::GetThresholdY() const
		{
			return afPlatform3Sigma[thisPlatform][1] * tsConfig.fThresholdMultiplier;
		}

		bool TouchDevice::InBox(const float &x0, const float &y0, const float &x1, const float &y1) const
		{
			return state.InBox(x0, y0, x1, y1);
		}
		bool TouchDevice::InBox(const Rectangle &bounds) const
		{
			return state.InBox(bounds);
		}

		bool TouchDevice::GestureDrag() const
		{
			if (state.endTime <= state.startTime)
				return false;

			return state.endTime - state.startTime > tsConfig.fSingleTouchMaxDuration;
		}

		bool TouchDevice::Calibrate()
		{
			if (dqSamples->Samples() < 2)
				return false;

			char buffer[200];
			static int filesCount = 0;
			sprintf(buffer, "calibration-%.2f-%02d.txt", tsConfig.fInputTimeFrame, filesCount);

			FILE *fp = fopen(buffer, "wb");
			if (fp == 0)
				return false;

			filesCount++;

			unsigned int samples = 0;
			unsigned int index = dqSamples->Front();
			while (true)
			{
				TouchSampleData &sample = dqSamples->Get(index);

				sprintf(buffer, "%.3f,%.3f,%.3f\n", sample.x, sample.y, sample.time);
				fwrite(buffer, 1 , strlen(buffer) , fp);

				motion[samples].dx = sample.x;
				motion[samples].dy = sample.y;
				samples++;

				if (index == dqSamples->Back())
					break;

				index = dqSamples->Prev(index);
			}
			MeanAndVar(samples, state.meanX, state.meanY, state.stdX, state.stdY);
			sprintf(buffer, "Calibration Statistics:\nSamples = %d\nMean = (%.6f,%.6f)\nStd  = (%.6f,%.6f)\n", samples, state.meanX, state.meanY, state.stdX, state.stdY);
			DebugPrint(buffer);
			fwrite(buffer, 1 , strlen(buffer) , fp);
			fclose(fp);

			return true;
		}

		/********************************************************************
		 * TouchState Implementation
		 ********************************************************************/

		bool TouchState::IsPressing() const
		{
			return pressing;
		}
		bool TouchState::IsDragging() const
		{
			return dragging;
		}
		bool TouchState::IsPressed() const
		{
			return pressed;
		}
		bool TouchState::IsReleased() const
		{
			return released;
		}

		bool TouchState::SingleTouch(float fSingleTouchMaxOffset, float fSingleTouchMinDuration, float fSingleTouchMaxDuration) const
		{
			if (!released)
				return false;

			float dx = endPosX - startPosX;
			float dy = endPosY - startPosY;

			bool bRet = sqrt(dx * dx + dy * dy) < fSingleTouchMaxOffset && endTime - startTime > fSingleTouchMinDuration && endTime - startTime < fSingleTouchMaxDuration;

			return bRet;
		}

		bool TouchState::InBox(const float &x0, const float &y0, const float &x1, const float &y1) const
		{
			return endPosX >= x0 && endPosX <= x1 && endPosY >= y0 && endPosY <= y1;
		}
		bool TouchState::InBox(const Rectangle &bounds) const
		{
			return InBox(bounds.left, bounds.bottom, bounds.right, bounds.top);
		}

		bool TouchState::QuitEvent()
		{
			bool bRet = quit;
			quit = false;
			return bRet;
		}
	}
}
