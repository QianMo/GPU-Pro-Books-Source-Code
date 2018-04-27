/*!****************************************************************************
 @File          OGLES2Initialization.mm
 
 @Title         OpenGL ES 2.0 Initialization Tutorial
 
 @Copyright     Copyright 2003-2004 by Imagination Technologies Limited.
 
 @Platform      iPhone
 
 @Description   Basic Tutorial that shows step-by-step how to initialize
				OpenGL ES 2.0, use it for clearing the screen with
				different colours and terminate it.

				This tutorial contains a platform specific part that may need
				modification before you can run it on a certain platform.
				
				Important resources for OpenGL ES 2.0 development can be found at 
				http://www.khronos.org/opengles/
 ******************************************************************************/
#import <QuartzCore/QuartzCore.h>
#import <UIKit/UIKit.h>
#import <OpenGLES/EAGL.h>
#import <OpenGLES/EAGLDrawable.h>
#import <OpenGLES/ES2/gl.h>
#import <OpenGLES/ES2/glext.h>


/******************************************************************************
 Defines
 ******************************************************************************/

#define KFPS			120.0


/*!****************************************************************************
 @Function		TestError
 @Input			pszLocation		location in the program where the error took
								place. ie: function name
 @Return		bool			true if no  error was detected
 @Description	Tests for an error and prints it
******************************************************************************/
bool TestError(const char* pszLocation)
{
	/*
		glGetError returns the last error that has happened using gl,
		not the status of the last called function. The user has to
		check after every single gl call or at least once every frame.
	*/
	GLint iErr = glGetError();
	if (iErr != GL_NO_ERROR)
	{
		printf("%s failed (%d).\n", pszLocation, iErr);
		return false;
	}

	return true;
}

/*!****************************************************************************
 Class EAGLView
*******************************************************************************/
@class EAGLView;

@interface EAGLView : UIView
{
@private
	EAGLContext			*m_context;
	GLuint				m_framebuffer;
	GLuint				m_renderbuffer;
	GLuint				m_depthBuffer;
	
	UIWindow*			m_window;
	NSTimer*			m_renderTimer;	
	
	int					m_iFrameCounter;
}

+ (Class) layerClass;
- (void) applicationDidFinishLaunching:(UIApplication*)application;
- (void) RenderScene;
- (void) dealloc;
@end



@implementation EAGLView

+ (Class) layerClass
{
	return [CAEAGLLayer class];
}

/*!****************************************************************************
 @Function		applicationDidFinishLaunching
 @Input			application		
 @Description	This method is called after the application has been loaded.
                We put our EAGL and OpenGL ES initialisation code here.
 ******************************************************************************/
- (void) applicationDidFinishLaunching:(UIApplication*)application
{
	
	/*
		Step 0 - Create a fullscreen window that we can use for OpenGL ES output
	*/
	CGRect rect = [[UIScreen mainScreen] bounds];	
	m_window = [[UIWindow alloc] initWithFrame:rect];
	
	if(!(self = [super initWithFrame:rect])) 
	{
		[self release];
		return;
	}
	
	/*
		Step 1 -Initialise EAGL.
	*/
	CAEAGLLayer* eaglLayer = (CAEAGLLayer*)[self layer];	
	[eaglLayer setDrawableProperties: [	NSDictionary dictionaryWithObjectsAndKeys: [NSNumber numberWithBool:NO], 
									   kEAGLDrawablePropertyRetainedBacking, 
									   kEAGLColorFormatRGBA8, 
									   kEAGLDrawablePropertyColorFormat, 
									   nil]];
	
	/*
		Step 2 - Create a context for rendering with OpenGL ES2.
	*/
	m_context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
	
	if((!m_context) || (![EAGLContext setCurrentContext:m_context])) 
	{
		[self release];
		return;
	}
	
	/*
		Step 3 - Get the window size.
	*/	
	CGSize	newSize;
	newSize = [eaglLayer bounds].size;
	newSize.width = roundf(newSize.width);
	newSize.height = roundf(newSize.height);
	
	/*
		Step 4 - Create a render buffer.
	*/
	GLuint oldRenderbuffer;
	glGetIntegerv(GL_RENDERBUFFER_BINDING, (GLint *) &oldRenderbuffer);
	glGenRenderbuffers(1, &m_renderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, m_renderbuffer);
	
	if(![m_context renderbufferStorage:GL_RENDERBUFFER fromDrawable:eaglLayer])
	{
		glDeleteRenderbuffers(1, &m_renderbuffer);
		glBindRenderbuffer(GL_RENDERBUFFER_BINDING, oldRenderbuffer);
		[self release];
		return;
	}
	
	/*
		Step 5 - Create a depth buffer.
	*/
	glGenRenderbuffers(1, &m_depthBuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, m_depthBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24_OES, newSize.width, newSize.height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depthBuffer);
	
	/*
		Step 6 - Create a frame buffer.
	*/	
	GLuint oldFramebuffer;
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, (GLint *) &oldFramebuffer);
	glGenFramebuffers(1, &m_framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer);
	
	/*
		Step 7 - Attach the render buffer to the framebuffer.
	*/	
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_renderbuffer);
	
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		[self release];
		return;
	}
	
	/*
		Step 9 - Set the viewport size to the window size.
	*/
	glViewport(0, 0, newSize.width, newSize.height);
	
	/*
		Step 10 - Add this view to the window.
	*/
	[m_window addSubview: self];
	
	
	
	/*
		Step 10 - Show the window.
	*/
	[m_window makeKeyAndVisible];
	
	
	/*
		Step 11 - Setup a timer to call RenderScene to actually draw the scene.
	*/
	[UIApplication sharedApplication].idleTimerDisabled = YES;
	m_renderTimer = [NSTimer scheduledTimerWithTimeInterval:(1.0 / KFPS) target:self selector:@selector(RenderScene) userInfo:nil repeats:YES];	
	
	m_iFrameCounter = 0;
	
}

/*!****************************************************************************
 @Function		RenderScene
 @Description	This method actually draws the scene. It gets called by our timer.
 ******************************************************************************/
- (void) RenderScene
{
	// Sets the clear color to a varying color.
	// The colours are passed per channel (red,green,blue,alpha) as float values from 0.0 to 1.0
	if (m_iFrameCounter & 128)
	{
		glClearColor(0.6f, 0.8f, 1.0f, 1.0f); // clear blue 
	}
	else
	{
		glClearColor(1.0f, 1.0f, 0.66f, 1.0f); // clear yellow
	}	

	/*
		Clears the color buffer.
		glClear() can also be used to clear the depth or stencil buffer
		(GL_DEPTH_BUFFER_BIT or GL_STENCIL_BUFFER_BIT)
	*/
	glClear(GL_COLOR_BUFFER_BIT);
	
	/*
		Swap Buffers.
		Brings to the native display the current render surface.
	*/
	EAGLContext *oldContext = [EAGLContext currentContext];
	GLuint oldRenderbuffer;
	
	if(oldContext != m_context)
		[EAGLContext setCurrentContext:m_context];
	
	
	glGetIntegerv(GL_RENDERBUFFER_BINDING, (GLint *) &oldRenderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, m_renderbuffer);
	
	if(![m_context presentRenderbuffer:GL_RENDERBUFFER])
		printf("Failed to swap renderbuffer in %s\n", __FUNCTION__);
	
	if(oldContext != m_context)
		[EAGLContext setCurrentContext:oldContext];
	
	++m_iFrameCounter;		
	if(m_iFrameCounter == 10000) m_iFrameCounter = 0;
}


/*!****************************************************************************
 @Function		dealloc
 @Description	Releases allocaed resources.
 ******************************************************************************/
- (void) dealloc
{
	/*
		Step 12 - Release buffers
	*/
	[m_window release];
	
	EAGLContext *oldContext = [EAGLContext currentContext];
	
	if (oldContext != m_context)
		[EAGLContext setCurrentContext:m_context];
	
	glDeleteRenderbuffers(1, &m_depthBuffer);
	m_depthBuffer = 0;
	
	
	glDeleteRenderbuffers(1, &m_renderbuffer);
	m_renderbuffer = 0;
	
	glDeleteFramebuffers(1, &m_framebuffer);	
	
	m_framebuffer = 0;
	if (oldContext != m_context)
		[EAGLContext setCurrentContext:oldContext];
	
	[m_context release];
	m_context = nil;
	
	[super dealloc];
}


@end

/*!****************************************************************************
 @Function		main
 @Description	Runs the application.
 ******************************************************************************/
int main(int argc, char **argv)
{
	NSAutoreleasePool* pool = [NSAutoreleasePool new];
	UIApplicationMain(argc, argv, nil, @"EAGLView");
	[pool release];
	return 0;
}
