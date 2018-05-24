#ifndef WGL_EXTENSIONS_H
#define WGL_EXTENSIONS_H

//
// - GLEW OpenGL-Extension Library does not init wgl-Extensions
// ->manually get function-pointers of wgl-functions
//
extern PFNWGLGETEXTENSIONSSTRINGARBPROC    wglGetExtensionsStringARB;
extern PFNWGLSWAPINTERVALEXTPROC          wglSwapIntervalEXT;
extern PFNWGLGETSWAPINTERVALEXTPROC       wglGetSwapIntervalEXT;

bool InitWGLExtensions(HDC hdc);
bool IsExtensionSupported(const char *wglExtensions, const char *extensionName);

#endif

