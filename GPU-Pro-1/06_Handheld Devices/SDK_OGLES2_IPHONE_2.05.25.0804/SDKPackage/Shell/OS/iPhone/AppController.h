/******************************************************************************

 @File         AppController.h

 @Title        

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     

 @Description  

******************************************************************************/
#ifndef _APPCONTROLLER_H_
#define _APPCONTROLLER_H_

#include "PVRShell.h"
#import "EAGLView.h"

//CLASS INTERFACES:

@interface AppController : NSObject <UIAccelerometerDelegate>
{
	UIWindow*				_window;
	EAGLView*				_glView;  // A view for OpenGL ES rendering
	NSTimer*				_renderTimer;	// timer for render loop
	UIAccelerationValue		_accelerometer[3];
	
	PVRShell*			m_pPVRShell;
	PVRShellInit*		m_pPVRShellInit ;
}

- (void) doExitFromFunction:(NSString*)reason;

@end

#endif _APPCONTROLLER_H_
