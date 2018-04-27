/******************************************************************************

 @File         EAGLView.h

 @Title        

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     

 @Description  

******************************************************************************/
#ifndef _EAGLVIEW_H_
#define _EAGLVIEW_H_


#import <UIKit/UIKit.h>
#import <OpenGLES/EAGL.h>
#import <OpenGLES/EAGLDrawable.h>


#ifdef BUILD_OGLES		
#import <OpenGLES/ES1/gl.h>
#import <OpenGLES/ES1/glext.h>
#elif BUILD_OGLES2
#import <OpenGLES/ES2/gl.h>
#import <OpenGLES/ES2/glext.h>
#endif

#include "PVRShell.h"

//CONSTANTS
const double c_TouchDistanceThreshold = 20.0;

//CLASSES:

@class EAGLView;

//PROTOCOLS:

@protocol EAGLViewDelegate <NSObject>
- (void) didResizeEAGLSurfaceForView:(EAGLView*)view; //Called whenever the EAGL surface has been resized
@end

//CLASS INTERFACE:

/*
This class wraps the CAEAGLLayer from CoreAnimation into a convenient UIView subclass.
The view content is basically an EAGL surface you render your OpenGL scene into.
Note that setting the view non-opaque will only work if the EAGL surface has an alpha channel.
*/
@interface EAGLView : UIView
{
@private
	NSString*				_format;
	GLuint					_depthFormat;
	GLuint					_stencilFormat;
	BOOL					_autoresize;
	EAGLContext				*_context;
	GLuint					_framebuffer;
	GLuint					_renderbuffer;
	GLuint					_depthBuffer;
	GLuint					_stencilBuffer;
	CGSize					_size;
	BOOL					_hasBeenCurrent;
	id<EAGLViewDelegate>	_delegate;
	
	CGPoint					m_TouchStart;
	PVRShellInit*			m_pPVRShellInit ;
}
- (id) initWithFrame:(CGRect)frame; //These also set the current context
- (id) initWithFrame:(CGRect)frame pixelFormat:(NSString*)format;
- (id) initWithFrame:(CGRect)frame pixelFormat:(NSString*)format depthFormat:(GLuint)depth stencilFormat:(GLuint)stencil preserveBackbuffer:(BOOL)retained;

@property(readonly) GLuint framebuffer;
@property(readonly) NSString* pixelFormat;
@property(readonly) GLuint depthFormat;
@property(readonly) EAGLContext *context;

@property BOOL autoresizesSurface; //NO by default - Set to YES to have the EAGL surface automatically resized when the view bounds change, otherwise the EAGL surface contents is rendered scaled
@property(readonly, nonatomic) CGSize surfaceSize;

@property(assign) id<EAGLViewDelegate> delegate;

- (void) setCurrentContext;
- (BOOL) isCurrentContext;
- (void) clearCurrentContext;

- (void) swapBuffers; //This also checks the current OpenGL error and logs an error if needed

- (CGPoint) convertPointFromViewToSurface:(CGPoint)point;
- (CGRect) convertRectFromViewToSurface:(CGRect)rect;

- (void) setPVRShellInit: (PVRShellInit*) pPVRShellInit;

@end

#endif // _EAGLVIEW_H_
