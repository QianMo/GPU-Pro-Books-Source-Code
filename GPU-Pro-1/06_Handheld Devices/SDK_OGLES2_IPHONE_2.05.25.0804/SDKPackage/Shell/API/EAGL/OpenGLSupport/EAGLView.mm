

#import <QuartzCore/QuartzCore.h>

#import "EAGLView.h"
#import "OpenGL_Internal.h"

#include "PVRShell.h"
#include "PVRShellOS.h"
#include "PVRShellAPI.h"
#include "PVRShellImpl.h"

//CLASS IMPLEMENTATIONS:

@implementation EAGLView

@synthesize delegate=_delegate, autoresizesSurface=_autoresize, surfaceSize=_size, framebuffer = _framebuffer, pixelFormat = _format, depthFormat = _depthFormat, context = _context;

+ (Class) layerClass
{
	return [CAEAGLLayer class];
}

- (BOOL) _createSurface
{
	CAEAGLLayer*			eaglLayer = (CAEAGLLayer*)[self layer];
	CGSize					newSize;
	GLuint					oldRenderbuffer;
	GLuint					oldFramebuffer;
	
	if(![EAGLContext setCurrentContext:_context]) {
		return NO;
	}
	
	newSize = [eaglLayer bounds].size;
	newSize.width = roundf(newSize.width);
	newSize.height = roundf(newSize.height);
	
#ifdef BUILD_OGLES		
	glGetIntegerv(GL_RENDERBUFFER_BINDING_OES, (GLint *) &oldRenderbuffer);
	glGetIntegerv(GL_FRAMEBUFFER_BINDING_OES, (GLint *) &oldFramebuffer);
	
	glGenRenderbuffersOES(1, &_renderbuffer);
	glBindRenderbufferOES(GL_RENDERBUFFER_OES, _renderbuffer);
	
	if(![_context renderbufferStorage:GL_RENDERBUFFER_OES fromDrawable:eaglLayer]) {
		glDeleteRenderbuffersOES(1, &_renderbuffer);
		glBindRenderbufferOES(GL_RENDERBUFFER_BINDING_OES, oldRenderbuffer);
		return NO;
	}
	
	glGenFramebuffersOES(1, &_framebuffer);
	glBindFramebufferOES(GL_FRAMEBUFFER_OES, _framebuffer);
	glFramebufferRenderbufferOES(GL_FRAMEBUFFER_OES, GL_COLOR_ATTACHMENT0_OES, GL_RENDERBUFFER_OES, _renderbuffer);
	if (_depthFormat)
	{
		glGenRenderbuffersOES(1, &_depthBuffer);
		glBindRenderbufferOES(GL_RENDERBUFFER_OES, _depthBuffer);
		glRenderbufferStorageOES(GL_RENDERBUFFER_OES, _depthFormat, newSize.width, newSize.height);
		glFramebufferRenderbufferOES(GL_FRAMEBUFFER_OES, GL_DEPTH_ATTACHMENT_OES, GL_RENDERBUFFER_OES, _depthBuffer);
	}
	
	_size = newSize;
	if(!_hasBeenCurrent)
	{
		glViewport(0, 0, newSize.width, newSize.height);
		glScissor(0, 0, newSize.width, newSize.height);
		_hasBeenCurrent = YES;
	}
	else
	{
		glBindFramebufferOES(GL_FRAMEBUFFER_OES, oldFramebuffer);
	}
	glBindRenderbufferOES(GL_RENDERBUFFER_OES, oldRenderbuffer);
#elif BUILD_OGLES2
	glGetIntegerv(GL_RENDERBUFFER_BINDING, (GLint *) &oldRenderbuffer);
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, (GLint *) &oldFramebuffer);
	
	glGenRenderbuffers(1, &_renderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, _renderbuffer);
	
	if(![_context renderbufferStorage:GL_RENDERBUFFER fromDrawable:eaglLayer])
	{
		glDeleteRenderbuffers(1, &_renderbuffer);
		glBindRenderbuffer(GL_RENDERBUFFER_BINDING, oldRenderbuffer);
		return NO;
	}
	
	glGenFramebuffers(1, &_framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, _framebuffer);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, _renderbuffer);
	if (_depthFormat)
	{
		if(_stencilFormat)
		{
			glGenRenderbuffers(1, &_depthBuffer);
			glBindRenderbuffer(GL_RENDERBUFFER, _depthBuffer);
			glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8_OES, newSize.width, newSize.height);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _depthBuffer);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, _depthBuffer);
		}
		else
		{
			glGenRenderbuffers(1, &_depthBuffer);
			glBindRenderbuffer(GL_RENDERBUFFER, _depthBuffer);
			glRenderbufferStorage(GL_RENDERBUFFER, _depthFormat, newSize.width, newSize.height);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _depthBuffer);
		}
	}

	GLuint status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE)
	{
		return NO;
	}
	
	_size = newSize;
	if(!_hasBeenCurrent) {
		glViewport(0, 0, newSize.width, newSize.height);
		glScissor(0, 0, newSize.width, newSize.height);
		_hasBeenCurrent = YES;
	}
	else {
		glBindFramebuffer(GL_FRAMEBUFFER, oldFramebuffer);
	}
	glBindRenderbuffer(GL_RENDERBUFFER, oldRenderbuffer);
#endif
	
	CHECK_GL_ERROR();
	
	[_delegate didResizeEAGLSurfaceForView:self];
	
	return YES;
}

- (void) _destroySurface
{
	EAGLContext *oldContext = [EAGLContext currentContext];
	
	if (oldContext != _context)
		[EAGLContext setCurrentContext:_context];

#ifdef BUILD_OGLES
	if(_depthFormat) {
		glDeleteRenderbuffersOES(1, &_depthBuffer);
		_depthBuffer = 0;
	}
	
	glDeleteRenderbuffersOES(1, &_renderbuffer);
	_renderbuffer = 0;

	glDeleteFramebuffersOES(1, &_framebuffer);
#elif BUILD_OGLES2
	if(_depthFormat) {
		glDeleteRenderbuffers(1, &_depthBuffer);
		_depthBuffer = 0;
	}
	
	glDeleteRenderbuffers(1, &_renderbuffer);
	_renderbuffer = 0;
	
	glDeleteFramebuffers(1, &_framebuffer);	
#endif
	_framebuffer = 0;
	if (oldContext != _context)
		[EAGLContext setCurrentContext:oldContext];
}

- (id) initWithFrame:(CGRect)frame
{
	return [self initWithFrame:frame pixelFormat:kEAGLColorFormatRGB565 depthFormat:0 stencilFormat:0 preserveBackbuffer:NO];
}

- (id) initWithFrame:(CGRect)frame pixelFormat:(NSString*)format 
{
	return [self initWithFrame:frame pixelFormat:format depthFormat:0 stencilFormat:0 preserveBackbuffer:NO];
}

- (id) initWithFrame:(CGRect)frame
		 pixelFormat:(NSString*)format
		 depthFormat:(GLuint)depth
	   stencilFormat:(GLuint)stencil
  preserveBackbuffer:(BOOL)retained
{
	if((self = [super initWithFrame:frame])) {
		CAEAGLLayer*			eaglLayer = (CAEAGLLayer*)[self layer];

//		[eaglLayer setDrawableProperties:[NSDictionary dictionaryWithObjectsAndKeys:[NSNumber numberWithBool:retained], kEAGLDrawablePropertyRetainedBacking, format, kEAGLDrawablePropertyColorFormat, nil]];
		[eaglLayer setDrawableProperties:[NSDictionary dictionaryWithObjectsAndKeys:[NSNumber numberWithBool:retained], kEAGLDrawablePropertyRetainedBacking, format, kEAGLDrawablePropertyColorFormat, nil]];
		_format = format;
		_depthFormat = depth;
		_stencilFormat = stencil;
		
#ifdef BUILD_OGLES		
		_context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES1];
#elif BUILD_OGLES2
		_context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
#endif
		
		if(_context == nil) {
			[self release];
			return nil;
		}
		
		if(![self _createSurface]) {
			[self release];
			return nil;
		}
		
		[self setMultipleTouchEnabled:YES];
	}

	return self;
}

- (void) dealloc
{
	[self _destroySurface];
	
	[_context release];
	_context = nil;
	
	[super dealloc];
}

- (void) layoutSubviews
{
	CGRect				bounds = [self bounds];
	
	if(_autoresize && ((roundf(bounds.size.width) != _size.width) || (roundf(bounds.size.height) != _size.height))) {
		[self _destroySurface];
#if __DEBUG__
		REPORT_ERROR(@"Resizing surface from %fx%f to %fx%f", _size.width, _size.height, roundf(bounds.size.width), roundf(bounds.size.height));
#endif
		[self _createSurface];
	}
}

- (void) setAutoresizesEAGLSurface:(BOOL)autoresizesEAGLSurface;
{
	_autoresize = autoresizesEAGLSurface;
	if(_autoresize)
	[self layoutSubviews];
}

- (void) setCurrentContext
{
	if(![EAGLContext setCurrentContext:_context]) {
		printf("Failed to set current context %p in %s\n", _context, __FUNCTION__);
	}
}

- (BOOL) isCurrentContext
{
	return ([EAGLContext currentContext] == _context ? YES : NO);
}

- (void) clearCurrentContext
{
	if(![EAGLContext setCurrentContext:nil])
		printf("Failed to clear current context in %s\n", __FUNCTION__);
}

- (void) swapBuffers
{
	EAGLContext *oldContext = [EAGLContext currentContext];
	GLuint oldRenderbuffer;
	
	if(oldContext != _context)
		[EAGLContext setCurrentContext:_context];
	
	CHECK_GL_ERROR();
	
#ifdef BUILD_OGLES
	glGetIntegerv(GL_RENDERBUFFER_BINDING_OES, (GLint *) &oldRenderbuffer);
	glBindRenderbufferOES(GL_RENDERBUFFER_OES, _renderbuffer);
	
	if(![_context presentRenderbuffer:GL_RENDERBUFFER_OES])
		printf("Failed to swap renderbuffer in %s\n", __FUNCTION__);
#elif BUILD_OGLES2
	glGetIntegerv(GL_RENDERBUFFER_BINDING, (GLint *) &oldRenderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, _renderbuffer);
	
	if(![_context presentRenderbuffer:GL_RENDERBUFFER])
		printf("Failed to swap renderbuffer in %s\n", __FUNCTION__);
#endif
	
	if(oldContext != _context)
		[EAGLContext setCurrentContext:oldContext];
}

- (CGPoint) convertPointFromViewToSurface:(CGPoint)point
{
	CGRect				bounds = [self bounds];
	
	return CGPointMake((point.x - bounds.origin.x) / bounds.size.width * _size.width, (point.y - bounds.origin.y) / bounds.size.height * _size.height);
}

- (CGRect) convertRectFromViewToSurface:(CGRect)rect
{
	CGRect				bounds = [self bounds];
	
	return CGRectMake((rect.origin.x - bounds.origin.x) / bounds.size.width * _size.width, (rect.origin.y - bounds.origin.y) / bounds.size.height * _size.height, rect.size.width / bounds.size.width * _size.width, rect.size.height / bounds.size.height * _size.height);
}

// Handles the start of a touch
- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event
{
	
	for( UITouch *touch in touches )
	{
		// Grab the location of the touch
		m_TouchStart = [touch locationInView:self];
		
		m_pPVRShellInit->m_bPointer = true;
		m_pPVRShellInit->m_bNormalized = false;

		m_pPVRShellInit->m_vec2PointerLocation[0] = m_TouchStart.x;
		m_pPVRShellInit->m_vec2PointerLocation[1] = m_TouchStart.y;

	}
}

// Handles the continuation of a touch.
- (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event
{
	
	for( UITouch *touch in touches )
	{
		// Grab the location of the touch
		CGPoint touchLocation = [touch locationInView:self];

		m_pPVRShellInit->m_bPointer = true;
		m_pPVRShellInit->m_bNormalized = false;

		m_pPVRShellInit->m_vec2PointerLocation[0] = touchLocation.x;
		m_pPVRShellInit->m_vec2PointerLocation[1] = touchLocation.y;

	}
}

// Handles the end of a touch event.
- (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event
{
	
	for( UITouch *touch in touches )
	{
		// Grab the location of the touch
		// look at start coords and finish coords and decide what action to set
		CGPoint touchLocation = [touch locationInView:self];
		
		CGPoint touchMove;	// no operator because we're in Obj-C
		touchMove.x = touchLocation.x - m_TouchStart.x;
		touchMove.y = touchLocation.y - m_TouchStart.y;

		m_pPVRShellInit->m_bPointer = false;
		m_pPVRShellInit->m_bNormalized = false;

		m_pPVRShellInit->m_vec2PointerLocation[0] = touchLocation.x;
		m_pPVRShellInit->m_vec2PointerLocation[1] = touchLocation.y;

		
		if(touchMove.x*touchMove.x + touchMove.y*touchMove.y<c_TouchDistanceThreshold)
		{
			m_pPVRShellInit->KeyPressed(PVRShellKeyNameACTION1);
		}
		else if (touchMove.x> c_TouchDistanceThreshold)
		{
			m_pPVRShellInit->KeyPressed(m_pPVRShellInit->m_eKeyMapUP);
		}
		else if (touchMove.x< -c_TouchDistanceThreshold)
		{
			m_pPVRShellInit->KeyPressed(m_pPVRShellInit->m_eKeyMapDOWN);
		}
		else if (touchMove.y> c_TouchDistanceThreshold)
		{
			m_pPVRShellInit->KeyPressed(m_pPVRShellInit->m_eKeyMapLEFT);
		}
		else if (touchMove.y< -c_TouchDistanceThreshold)
		{
			m_pPVRShellInit->KeyPressed(m_pPVRShellInit->m_eKeyMapRIGHT);
		}
		
	}
}

- (void) setPVRShellInit: (PVRShellInit *)pPVRShellInit
{
	m_pPVRShellInit = pPVRShellInit;
}

@end
