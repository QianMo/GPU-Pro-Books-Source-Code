#import "AppController.h"

#include "PVRShellAPI.h"
#include "PVRShellOS.h"
#include "PVRShellImpl.h"

//CONSTANTS:

const int kFPS = 120.0;
#define kAccelerometerFrequency		30.0 //Hz
#define kFilteringFactor			0.1

// MACROS
#define DEGREES_TO_RADIANS(__ANGLE__) ((__ANGLE__) / 180.0 * M_PI)

// CLASS IMPLEMENTATION
@implementation AppController

- (void) _renderGLScene
{
	// TODO take bool from this and signal exit if appropriate
	if(!m_pPVRShell->RenderScene())
	{
		[self doExitFromFunction:@"RenderScene() returned false. Exiting...\n"];
		[_renderTimer invalidate];
		[_renderTimer release];
		[[UIApplication sharedApplication] performSelector:@selector(terminateWithSuccess)];
	}
	
	//Swap framebuffer
	[_glView swapBuffers];
}

- (void) applicationDidFinishLaunching:(UIApplication*)application
{
	CGRect					rect = [[UIScreen mainScreen] bounds];	

	m_pPVRShellInit = new PVRShellInit;
	m_pPVRShell = NewDemo();
	
	if(!m_pPVRShell)
	{
		[self doExitFromFunction:@"NewDemo() Failed\n"];
		return;
	}
	
	m_pPVRShellInit->Init(*m_pPVRShell);
	
	// fake command line input
	// if your application is expecting command line input then fake it here
	char pszCL[] = "";
	m_pPVRShellInit->CommandLine(pszCL);
	
	// set up file paths
	NSString* readPath = [NSString stringWithFormat:@"%@%@", [[NSBundle mainBundle] bundlePath], @"/"];
	m_pPVRShellInit->SetReadPath([readPath UTF8String]);

  	NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    	NSString *documentsDirectory = [paths objectAtIndex:0];
	m_pPVRShellInit->SetWritePath([documentsDirectory UTF8String]);
	
	
	if(m_pPVRShell->InitApplication())
	{
		printf("InitApplication() succeeded\n");
	}
	else
	{	
		[self doExitFromFunction:@"InitApplication() failed"];
		return;
	}
	
	//Create a full-screen window
	_window = [[UIWindow alloc] initWithFrame:rect];
	
	NSString *strColourFormat;
	int iDepthFormat, iStencilFormat;
	
	if(m_pPVRShell->PVRShellGet(prefColorBPP)==16)
	{
		strColourFormat = kEAGLColorFormatRGB565;
	}
	else
	{
		strColourFormat = kEAGLColorFormatRGBA8;
	}

	
#ifdef BUILD_OGLES2
	//Create the OpenGL ES view and add it to the window
	if(m_pPVRShell->PVRShellGet(prefStencilBufferContext))
	{
		if(m_pPVRShell->PVRShellGet(prefZbufferContext))
		{	// need stencil & depth
			iDepthFormat = iStencilFormat = GL_DEPTH24_STENCIL8_OES;
		}
		else
		{	// just stencil
			iDepthFormat = 0;
			iStencilFormat = GL_STENCIL_INDEX8;
		}
	}
	else 
#endif
		
	if (m_pPVRShell->PVRShellGet(prefZbufferContext))
	{	// just depth buffer
			iDepthFormat = GL_DEPTH_COMPONENT24_OES;
			iStencilFormat = 0;
	}
	else 
	{	// neither depth nor stencil
		iDepthFormat = iStencilFormat = 0;
	}

	// actualy make it
	_glView = [[EAGLView alloc] initWithFrame:rect pixelFormat:strColourFormat depthFormat:iDepthFormat stencilFormat:iStencilFormat preserveBackbuffer:NO];

	[_window addSubview:_glView];

	[_glView setPVRShellInit:m_pPVRShellInit];
	
	if(m_pPVRShell->InitView())
	{
		printf("InitView() succeeded\n");
	}
	else
	{
		[self doExitFromFunction:@"InitView() Failed\n"];
		return;
	}

	//Show the window
	[_window makeKeyAndVisible];
	
	//Configure and start accelerometer
	[[UIAccelerometer sharedAccelerometer] setUpdateInterval:(1.0 / kAccelerometerFrequency)];
	[[UIAccelerometer sharedAccelerometer] setDelegate:self];

	[UIApplication sharedApplication].idleTimerDisabled = YES;
	
	//Render the initial frame
	[self _renderGLScene];
	
	//Create our rendering timer
	// TODO: find a way of doing this without the timer i.e. no refresh speed
	
	_renderTimer = [NSTimer scheduledTimerWithTimeInterval:(1.0 / kFPS) target:self selector:@selector(_renderGLScene) userInfo:nil repeats:YES];	
}

- (void) dealloc {
	// this doesn't seem to be called
	// TODO: work out if it needs to be called
	// work out how to get it called
	printf("Dealloc called\n");
	
	
	
	if(m_pPVRShell->ReleaseView())
	{
		printf("ReleaseView() succeeded\n");
	}
	else
	{
		[self doExitFromFunction:@"ReleaseView() Failed\n"];
		return;
	}

	[_glView release];
	[_window release];
	
	if(m_pPVRShell->QuitApplication())
	{
		printf("QuitApplication() succeeded\n");
	}
	else
	{
		[self doExitFromFunction:@"QuitApplication() Failed\n"];
		return;
	}

	[super dealloc];
}

// throws up a warning dialog
- (void) doExitFromFunction:(NSString*)reason
{
	// TODO get an OK button
	printf("%s\n",[reason UTF8String]);
	UIAlertView *myExitWindow = [[UIAlertView alloc] initWithFrame: [[UIScreen mainScreen] bounds]];		// TODO: exit and alert
	[myExitWindow setTitle:reason];
	if(m_pPVRShell->PVRShellGet(prefExitMessage))
	{	// if this message is unset then this avoids a crash
		[myExitWindow setMessage:[NSString stringWithCString:(const char*)m_pPVRShell->PVRShellGet(prefExitMessage)]];
	}
	else
	{
		[myExitWindow setMessage:@"Exit message is unset"];
	}
	[myExitWindow show];
	
}


- (void) accelerometer:(UIAccelerometer*)accelerometer didAccelerate:(UIAcceleration*)acceleration {
	//Use a basic low-pass filter to only keep the gravity in the accelerometer values
	/*_accelerometer[0] = acceleration.x * kFilteringFactor + _accelerometer[0] * (1.0 - kFilteringFactor);
	_accelerometer[1] = acceleration.y * kFilteringFactor + _accelerometer[1] * (1.0 - kFilteringFactor);
	_accelerometer[2] = acceleration.z * kFilteringFactor + _accelerometer[2] * (1.0 - kFilteringFactor);*/
	
	m_pPVRShellInit->m_vec3Accel[0] = acceleration.x * kFilteringFactor + _accelerometer[0] * (1.0 - kFilteringFactor);
	m_pPVRShellInit->m_vec3Accel[1] = acceleration.y * kFilteringFactor + _accelerometer[1] * (1.0 - kFilteringFactor);
	m_pPVRShellInit->m_vec3Accel[2] = acceleration.z * kFilteringFactor + _accelerometer[2] * (1.0 - kFilteringFactor);
	

	//Render a frame
	//[self _renderGLScene];
}

@end
