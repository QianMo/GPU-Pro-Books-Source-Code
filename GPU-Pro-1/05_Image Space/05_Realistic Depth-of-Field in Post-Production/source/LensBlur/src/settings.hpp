/* ******************************************************************************
* Description: Application parameters.
*
*  Version 1.0.0
*  Date: Nov 22, 2008
*  Author: David Illes, Peter Horvath
*
* GPUPro
***************************************************************************** */

#define COMPILE_GPU 1

#define LOG_FILE "f:/work/gyar/test/medusa/pixel.log"
#define LOG_TIME 0
#define TRACE_PIXEL 0
#define PIXELX 626
#define PIXELY 288

// gpu settings
#define NUM_BLOCKS    2
#define NUM_THREADS   8			// max is 16x16 on GeForce 9800 GT

#define PIXEL_GROUPS 16
#define KERNEL_SIZE 4096

// default values
#define MODE_DEFAULT 0				// 0 (artist), 1 (physics), 2 (preview)
#define SAMPLING_DEFAULT 2			// 0 - 8
#define EPSILON_DEFAULT 0.1f		// 0.0 - 1.0
#define OVERLAP_DEFAULT 1.0f		// 0.0 - 2.0
#define ZMIN_DEFAULT 0				// 0 - 255
#define ZMAX_DEFAULT 255			// 0 - 255
#define BLUR_DEFAULT 10				// kernel width (0-50)
#define INTERPOLATION_DEFAULT 0		// 0 / 1
#define APERTURE_DEFAULT 4			// kernel type (0-7)
#define FOCALPOINT_DEFAULT 255		// 0 - 255
#define FLENGTH_DEFAULT 50			// 18.0-600.0
#define MS_DEFAULT 1.0				// 0.5 - 4.0
#define FSTOP_DEFAULT 4				// 0 - 12 
#define DISTANCE_DEFAULT 4			// 0 - 200 (m) 
#define CAMERA_DISTANCE_DEFAULT 2	// 0 - 200 (m) 
#define BLOOMTHRESHOLD_DEFAULT 255	// 0 - 255
#define BLOOMAMOUNT_DEFAULT 0		// 0 - 100
