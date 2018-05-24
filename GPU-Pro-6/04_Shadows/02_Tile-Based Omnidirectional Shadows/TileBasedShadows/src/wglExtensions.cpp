#include <stdafx.h>
#include <wglExtensions.h>

PFNWGLGETEXTENSIONSSTRINGARBPROC    wglGetExtensionsStringARB = NULL;
PFNWGLSWAPINTERVALEXTPROC	          wglSwapIntervalEXT = NULL;
PFNWGLGETSWAPINTERVALEXTPROC	      wglGetSwapIntervalEXT = NULL;

bool InitWGLExtensions(HDC hdc)
{
  // try to get function-pointer for retrieving wglExtension-string
  // if succeeded, retrieve wglExtension-String
  wglGetExtensionsStringARB = (PFNWGLGETEXTENSIONSSTRINGARBPROC) wglGetProcAddress("wglGetExtensionsStringARB");
  if(!wglGetExtensionsStringARB)
    return false; 
  const char *wglExtensions;
  wglExtensions = wglGetExtensionsStringARB(hdc);

  // check, if swap-control for toggling vertical synchronization is supported
  // if supported, get corresponding function-pointers
  if(!IsExtensionSupported(wglExtensions, "WGL_EXT_swap_control"))
    return false;     
  wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC) wglGetProcAddress("wglSwapIntervalEXT");
  wglGetSwapIntervalEXT = (PFNWGLGETSWAPINTERVALEXTPROC) wglGetProcAddress("wglGetSwapIntervalEXT");

  return true;
}

bool IsExtensionSupported(const char *wglExtensions, const char *extensionName)
{
  // check if extensionName is part of wglExtension-string
  return (strstr(wglExtensions, extensionName) != NULL);
}


