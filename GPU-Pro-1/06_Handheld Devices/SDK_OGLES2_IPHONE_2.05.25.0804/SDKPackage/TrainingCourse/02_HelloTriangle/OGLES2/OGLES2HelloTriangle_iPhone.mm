/*!****************************************************************************
 @File          OGLES2HelloTriangle_iPhone.m
 
 @Title         OpenGL ES 2.0 Hello Triangle Tutorial
 
 @Copyright     Copyright 2003-2004 by Imagination Technologies Limited.
 
 @Platform      iPhone
 
 @Description   Basic Tutorial that shows step-by-step how to initialize
 OpenGL ES 2.0, use it for drawing a triangle and terminate it.
 
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

// Index to bind the attributes to vertex shaders
#define VERTEX_ARRAY	0
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
	
	GLuint				m_uiVertexShader, m_uiFragShader;
	GLuint				m_uiProgramObject;
	GLuint				m_ui32Vbo;	
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
	// Fragment and vertex shaders code
	const char* pszFragShader = "\
	void main (void)\
	{\
	gl_FragColor = vec4(1.0, 1.0, 0.66  ,1.0);\
	}";
	
	const char* pszVertShader = "\
	attribute highp vec4	myVertex;\
	uniform mediump mat4	myPMVMatrix;\
	void main(void)\
	{\
	gl_Position = myPMVMatrix * myVertex;\
	}";	
	
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
		Step 11 - Draw something with OpenGL ES.
		At this point everything is initialized and we're ready to use
		OpenGL ES to draw something on the screen.
	*/	
	{
		// Create the fragment shader object
		m_uiFragShader = glCreateShader(GL_FRAGMENT_SHADER);
		
		// Load the source code into it
		glShaderSource(m_uiFragShader, 1, (const char**)&pszFragShader, NULL);
		
		// Compile the source code
		glCompileShader(m_uiFragShader);
		
		// Check if compilation succeeded
		GLint bShaderCompiled;
		glGetShaderiv(m_uiFragShader, GL_COMPILE_STATUS, &bShaderCompiled);
		
		if (!bShaderCompiled)
		{
			// An error happened, first retrieve the length of the log message
			int i32InfoLogLength, i32CharsWritten;
			glGetShaderiv(m_uiFragShader, GL_INFO_LOG_LENGTH, &i32InfoLogLength);
			
			// Allocate enough space for the message and retrieve it
			char* pszInfoLog = (char*)calloc(i32InfoLogLength, sizeof(char));
			glGetShaderInfoLog(m_uiFragShader, i32InfoLogLength, &i32CharsWritten, pszInfoLog);
			
			// Displays the error
			printf("Failed to compile fragment shader: %s\n", pszInfoLog);
			free(pszInfoLog);
			[self release];
			return;
		}
		
		// Loads the vertex shader in the same way
		m_uiVertexShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(m_uiVertexShader, 1, (const char**)&pszVertShader, NULL);
		glCompileShader(m_uiVertexShader);
		glGetShaderiv(m_uiVertexShader, GL_COMPILE_STATUS, &bShaderCompiled);
		
		if (!bShaderCompiled)
		{
			int i32InfoLogLength, i32CharsWritten;
			glGetShaderiv(m_uiVertexShader, GL_INFO_LOG_LENGTH, &i32InfoLogLength);
			char* pszInfoLog = (char*)calloc(i32InfoLogLength, sizeof(char));
			glGetShaderInfoLog(m_uiVertexShader, i32InfoLogLength, &i32CharsWritten, pszInfoLog);
			printf("Failed to compile vertex shader: %s\n", pszInfoLog);
			free(pszInfoLog);
			[self release];
			return;
		}
		
		// Create the shader program
		m_uiProgramObject = glCreateProgram();
		
		// Attach the fragment and vertex shaders to it
		glAttachShader(m_uiProgramObject, m_uiFragShader);
		glAttachShader(m_uiProgramObject, m_uiVertexShader);
		
		// Bind the custom vertex attribute "myVertex" to location VERTEX_ARRAY
		glBindAttribLocation(m_uiProgramObject, VERTEX_ARRAY, "myVertex");
		
		// Link the program
		glLinkProgram(m_uiProgramObject);
		
		// Check if linking succeeded in the same way we checked for compilation success
		GLint bLinked;
		glGetProgramiv(m_uiProgramObject, GL_LINK_STATUS, &bLinked);
		
		if (!bLinked)
		{
			int i32InfoLogLength, i32CharsWritten;
			glGetProgramiv(m_uiProgramObject, GL_INFO_LOG_LENGTH, &i32InfoLogLength);
			char* pszInfoLog = (char*)calloc(i32InfoLogLength, sizeof(char));
			glGetProgramInfoLog(m_uiProgramObject, i32InfoLogLength, &i32CharsWritten, pszInfoLog);
			printf("Failed to link program: %s\n", pszInfoLog);
			free(pszInfoLog);
			[self release];
			return;
		}
		
		// Actually use the created program
		glUseProgram(m_uiProgramObject);
		
		// Sets the clear color.
		// The colours are passed per channel (red,green,blue,alpha) as float values from 0.0 to 1.0
		glClearColor(0.6f, 0.8f, 1.0f, 1.0f); // clear blue
		
		// We're going to draw a triangle to the screen so create a vertex buffer object for our triangle
		
		// Interleaved vertex data
		GLfloat afVertices[] = {	-0.4f,-0.4f,0.0f, // Position
									0.4f ,-0.4f,0.0f,
									0.0f ,0.4f ,0.0f };
		
		// Generate the vertex buffer object (VBO)
		glGenBuffers(1, &m_ui32Vbo);
		
		// Bind the VBO so we can fill it with data
		glBindBuffer(GL_ARRAY_BUFFER, m_ui32Vbo);
		
		// Set the buffer's data
		unsigned int uiSize = 3 * (sizeof(GLfloat) * 3); // Calc afVertices size (3 vertices * stride (3 GLfloats per vertex))
		glBufferData(GL_ARRAY_BUFFER, uiSize, afVertices, GL_STATIC_DRAW);
		
	}
	
	/*
		Step 11 - Show the window.
	*/
	[m_window makeKeyAndVisible];
	
	
	/*
		Step 12 - Setup a timer to call RenderScene to actually draw the scene.
	*/
	[UIApplication sharedApplication].idleTimerDisabled = YES;
	m_renderTimer = [NSTimer scheduledTimerWithTimeInterval:(1.0 / KFPS) target:self selector:@selector(RenderScene) userInfo:nil repeats:YES];	
	
}

/*!****************************************************************************
 @Function		RenderScene
 @Description	This method actually draws the scene. It gets called by our timer.
 ******************************************************************************/
- (void) RenderScene
{
	// Matrix used for projection model view (PMVMatrix)
	float pfIdentity[] =
	{
		1.0f,0.0f,0.0f,0.0f,
		0.0f,1.0f,0.0f,0.0f,
		0.0f,0.0f,1.0f,0.0f,
		0.0f,0.0f,0.0f,1.0f
	};
	
	/*
	 Clears the color buffer.
	 glClear() can also be used to clear the depth or stencil buffer
	 (GL_DEPTH_BUFFER_BIT or GL_STENCIL_BUFFER_BIT)
	 */
	glClear(GL_COLOR_BUFFER_BIT);
	
	/*
	 Bind the projection model view matrix (PMVMatrix) to
	 the associated uniform variable in the shader
	 */
	
	// First gets the location of that variable in the shader using its name
	int i32Location = glGetUniformLocation(m_uiProgramObject, "myPMVMatrix");
	
	// Then passes the matrix to that variable
	glUniformMatrix4fv(i32Location, 1, GL_FALSE, pfIdentity);
	
	// Bind the VBO
	glBindBuffer(GL_ARRAY_BUFFER, m_ui32Vbo);
	
	/*
	 Enable the custom vertex attribute at index VERTEX_ARRAY.
	 We previously binded that index to the variable in our shader "vec4 MyVertex;"
	 */
	glEnableVertexAttribArray(VERTEX_ARRAY);
	
	// Sets the vertex data to this attribute index
	glVertexAttribPointer(VERTEX_ARRAY, 3, GL_FLOAT, GL_FALSE, 0, 0);
	
	/*
	 Draws a non-indexed triangle array from the pointers previously given.
	 This function allows the use of other primitive types : triangle strips, lines, ...
	 For indexed geometry, use the function glDrawElements() with an index list.
	 */
	glDrawArrays(GL_TRIANGLES, 0, 3);
	
	// Unbind the VBO
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	
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
	
}


/*!****************************************************************************
 @Function		dealloc
 @Description	Releases allocaed resources.
 ******************************************************************************/
- (void) dealloc
{
	// Delete the VBO as it is no longer needed
	glDeleteBuffers(1, &m_ui32Vbo);
	
	// Frees the OpenGL handles for the program and the 2 shaders
	glDeleteProgram(m_uiProgramObject);
	glDeleteShader(m_uiVertexShader);
	glDeleteShader(m_uiFragShader);
	
	/*
		Step 13 - Release buffers
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
