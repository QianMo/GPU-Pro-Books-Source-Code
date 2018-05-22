
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "OpenGLApp.h"

OpenGLApp::OpenGLApp(){
	glContext = NULL;
}

#if defined(_WIN32)

#pragma comment (lib, "opengl32.lib")

void initEntryPoints(HWND hwnd, const PIXELFORMATDESCRIPTOR &pfd){
	HDC hdc = GetDC(hwnd);

	int nPixelFormat = ChoosePixelFormat(hdc, &pfd);
	SetPixelFormat(hdc, nPixelFormat, &pfd);

	HGLRC hglrc = wglCreateContext(hdc);
	wglMakeCurrent(hdc, hglrc);

	initExtensions(hdc);

	wglMakeCurrent(NULL, NULL);
	wglDeleteContext(hglrc);
	ReleaseDC(hwnd, hdc);
}

LRESULT CALLBACK PFWinProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam){
	return DefWindowProc(hwnd, message, wParam, lParam);
};

bool OpenGLApp::initCaps(){
	PIXELFORMATDESCRIPTOR pfd = {
        sizeof (PIXELFORMATDESCRIPTOR), 1,
		PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
		PFD_TYPE_RGBA, colorBits,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		depthBits, stencilBits,
		0, PFD_MAIN_PLANE, 0, 0, 0, 0
    };

	WNDCLASS wincl;
	wincl.hInstance = hInstance;
	wincl.lpszClassName = "PFrmt";
	wincl.lpfnWndProc = PFWinProc;
	wincl.style = 0;
	wincl.hIcon = NULL;
	wincl.hCursor = NULL;
	wincl.lpszMenuName = NULL;
	wincl.cbClsExtra = 0;
	wincl.cbWndExtra = 0;
	wincl.hbrBackground = NULL;
	RegisterClass(&wincl);

	HWND hPFwnd = CreateWindow("PFrmt", "PFormat", WS_POPUP | WS_CLIPCHILDREN | WS_CLIPSIBLINGS, 0, 0, 8, 8, HWND_DESKTOP, NULL, hInstance, NULL);
	initEntryPoints(hPFwnd, pfd);
	SendMessage(hPFwnd, WM_CLOSE, 0, 0);

	return true;
}

MONITORINFO monInfo;

BOOL CALLBACK MonitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData){
	if (*(int *) dwData == 0){
		monInfo.cbSize = sizeof(monInfo);
		GetMonitorInfo(hMonitor, &monInfo);
		return FALSE;
	}
	(*(int *) dwData)--;

	return TRUE;
}

bool OpenGLApp::initAPI(){
	if (screen >= GetSystemMetrics(SM_CMONITORS)) screen = 0;

	int monitorCounter = screen;
	EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, (LPARAM) &monitorCounter);

	DWORD flags = WS_CLIPCHILDREN | WS_CLIPSIBLINGS;
	int x, y;


	x = monInfo.rcMonitor.left;
	y = monInfo.rcMonitor.top;

	device.cb = sizeof(device);
	EnumDisplayDevices(NULL, screen, &device, 0);

	DEVMODE dm, tdm;
	memset(&dm, 0, sizeof(dm));
	dm.dmSize = sizeof(dm);
	dm.dmBitsPerPel = colorBits;
	dm.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY;
	dm.dmPelsWidth  = fullscreenWidth;
	dm.dmPelsHeight = fullscreenHeight;
	dm.dmDisplayFrequency = 60;

	// Find a suitable fullscreen format
	int i = 0;
	int targetHz = 85;
	char str[128];

	resolution->clear();
	while (EnumDisplaySettings((const char *) device.DeviceName, i, &tdm)){
		if (int(tdm.dmBitsPerPel) == colorBits && tdm.dmPelsWidth >= 640 && tdm.dmPelsHeight >= 480){
			sprintf(str, "%dx%d", tdm.dmPelsWidth, tdm.dmPelsHeight);
			int index = resolution->addItemUnique(str);

			if (int(tdm.dmPelsWidth) == fullscreenWidth && int(tdm.dmPelsHeight) == fullscreenHeight){
				if (abs(int(tdm.dmDisplayFrequency) - targetHz) < abs(int(dm.dmDisplayFrequency) - targetHz)){
					dm = tdm;
				}
				resolution->selectItem(index);
			}
		}
		i++;
	}


	if (fullscreen){
		if (ChangeDisplaySettingsEx((const char *) device.DeviceName, &dm, NULL, CDS_FULLSCREEN, NULL) == DISP_CHANGE_SUCCESSFUL){
			flags |= WS_POPUP;
			captureMouse(!configDialog->isVisible());
		} else {
			ErrorMsg("Couldn't set fullscreen mode");
			fullscreen = false;
		}
	}

	sprintf(str, "%s (%dx%d)", getTitle(), width, height);
	if (!fullscreen){
		flags |= WS_OVERLAPPEDWINDOW;

		RECT wRect;
		wRect.left = 0;
		wRect.right = width;
		wRect.top = 0;
		wRect.bottom = height;
		AdjustWindowRect(&wRect, flags, FALSE);

		width  = min(wRect.right  - wRect.left, monInfo.rcWork.right  - monInfo.rcWork.left);
		height = min(wRect.bottom - wRect.top,  monInfo.rcWork.bottom - monInfo.rcWork.top);

		x = (monInfo.rcWork.left + monInfo.rcWork.right  - width ) / 2;
		y = (monInfo.rcWork.top  + monInfo.rcWork.bottom - height) / 2;
	}


	hwnd = CreateWindow("Humus", str, flags, x, y, width, height, HWND_DESKTOP, NULL, hInstance, NULL);



	PIXELFORMATDESCRIPTOR pfd = {
        sizeof (PIXELFORMATDESCRIPTOR), 1,
		PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
		PFD_TYPE_RGBA, colorBits,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		depthBits, stencilBits,
		0, PFD_MAIN_PLANE, 0, 0, 0, 0
    };

	hdc = GetDC(hwnd);

	int iAttribs[] = {
		WGL_DRAW_TO_WINDOW_ARB, GL_TRUE,
		WGL_ACCELERATION_ARB,   WGL_FULL_ACCELERATION_ARB,
		WGL_DOUBLE_BUFFER_ARB,  GL_TRUE,
		WGL_RED_BITS_ARB,       8,
		WGL_GREEN_BITS_ARB,     8,
		WGL_BLUE_BITS_ARB,      8,
		WGL_ALPHA_BITS_ARB,     (colorBits > 24)? 8 : 0,
		WGL_DEPTH_BITS_ARB,     depthBits,
		WGL_STENCIL_BITS_ARB,   stencilBits,
		0
	};

	int pixelFormats[256];
	int bestFormat = 0;
	int bestSamples = 0;
	uint nPFormats;
	if (WGL_ARB_pixel_format_supported && wglChoosePixelFormatARB(hdc, iAttribs, NULL, elementsOf(pixelFormats), pixelFormats, &nPFormats) && nPFormats > 0){
		int minDiff = 0x7FFFFFFF;
		int attrib = WGL_SAMPLES_ARB;
		int samples;

		// Find a multisample format as close as possible to the requested
		for (uint i = 0; i < nPFormats; i++){
			wglGetPixelFormatAttribivARB(hdc, pixelFormats[i], 0, 1, &attrib, &samples);
			int diff = abs(antiAliasSamples - samples);
			if (diff < minDiff){
				minDiff = diff;
				bestFormat = i;
				bestSamples = samples;
			}
		}
	} else {
		pixelFormats[0] = ChoosePixelFormat(hdc, &pfd);
	}
	antiAliasSamples = bestSamples;

	SetPixelFormat(hdc, pixelFormats[bestFormat], &pfd);

	glContext = wglCreateContext(hdc);
	wglMakeCurrent(hdc, glContext);

	initExtensions(hdc);

	if (WGL_ARB_multisample_supported && GL_ARB_multisample_supported && antiAliasSamples > 0){
		glEnable(GL_MULTISAMPLE_ARB);
	}

	if (fullscreen) captureMouse(!configDialog->isVisible());

	renderer = new OpenGLRenderer(hdc, glContext);
	renderer->setViewport(width, height);

	antiAlias->selectItem(antiAliasSamples / 2);

	linearClamp = renderer->addSamplerState(LINEAR, CLAMP, CLAMP, CLAMP);
	defaultFont = renderer->addFont("../Textures/Fonts/Future.dds", "../Textures/Fonts/Future.font", linearClamp);
	blendSrcAlpha = renderer->addBlendState(SRC_ALPHA, ONE_MINUS_SRC_ALPHA);
	noDepthTest  = renderer->addDepthState(false, false);
	noDepthWrite = renderer->addDepthState(true,  false);
	cullNone  = renderer->addRasterizerState(CULL_NONE);
	cullBack  = renderer->addRasterizerState(CULL_BACK);
	cullFront = renderer->addRasterizerState(CULL_FRONT);

	return true;
}

void OpenGLApp::exitAPI(){
	delete renderer;

	wglMakeCurrent(NULL, NULL);
	wglDeleteContext(glContext);
	ReleaseDC(hwnd, hdc);

	DestroyWindow(hwnd);

	if (fullscreen){
		// Reset display mode to default
		ChangeDisplaySettingsEx((const char *) device.DeviceName, NULL, NULL, 0, NULL);
	}
}

#else


bool OpenGLApp::initCaps(){
	return true;
}

struct DispRes {
	int w, h, index;
};

struct DispRes newRes(const int w, const int h, const int i){
	DispRes dr = { w, h, i };
	return dr;
}
int dComp(const DispRes &d0, const DispRes &d1){
	if (d0.w != d1.w) return (d0.w - d1.w); else return (d0.h - d1.h);
}

#if defined(LINUX)

bool OpenGLApp::initAPI(){
	screen = DefaultScreen(display);

	int nModes;
    XF86VidModeGetAllModeLines(display, screen, &nModes, &dmodes);

	Array <DispRes> modes;

	char str[64];
	int foundMode = -1;
	for (int i = 0; i < nModes; i++){
		if (dmodes[i]->hdisplay >= 640 && dmodes[i]->vdisplay >= 480){
			modes.add(newRes(dmodes[i]->hdisplay, dmodes[i]->vdisplay, i));

			if (dmodes[i]->hdisplay == fullscreenWidth && dmodes[i]->vdisplay == fullscreenHeight){
				foundMode = i;
			}
		}
	}

	resolution->clear();
	modes.sort(dComp);
	for (uint i = 0; i < modes.getCount(); i++){
		sprintf(str, "%dx%d", modes[i].w, modes[i].h);
		int index = resolution->addItemUnique(str);
		if (modes[i].index == foundMode) resolution->selectItem(index);
	}


	if (fullscreen){
		if (foundMode >= 0 && XF86VidModeSwitchToMode(display, screen, dmodes[foundMode])){
			XF86VidModeSetViewPort(display, screen, 0, 0);
		} else {
			char str[128];
			sprintf(str, "Couldn't set fullscreen at %dx%d.", fullscreenWidth, fullscreenHeight);
			ErrorMsg(str);
			fullscreen = false;
		}
	}

	XVisualInfo *vi;
	while (true){
		int attribs[] = {
			GLX_RGBA,
			GLX_DOUBLEBUFFER,
			GLX_RED_SIZE,      8,
			GLX_GREEN_SIZE,    8,
			GLX_BLUE_SIZE,     8,
			GLX_ALPHA_SIZE,    (colorBits > 24)? 8 : 0,
			GLX_DEPTH_SIZE,    depthBits,
			GLX_STENCIL_SIZE,  stencilBits,
			GLX_SAMPLE_BUFFERS_ARB, (antiAliasSamples > 0),
			GLX_SAMPLES_ARB,         antiAliasSamples,
			None,
		};

		vi = glXChooseVisual(display, screen, attribs);
		if (vi != NULL) break;

		antiAliasSamples -= 2;
		if (antiAliasSamples < 0){
			char str[256];
			sprintf(str, "No Visual matching colorBits=%d, depthBits=%d and stencilBits=%d", colorBits, depthBits, stencilBits);
			ErrorMsg(str);
			return false;
		}
	}


	//printf("Selected visual = 0x%x\n",(unsigned int) (vi->visualid));
	glContext = glXCreateContext(display, vi, None, True);


	XSetWindowAttributes attr;
	attr.colormap = XCreateColormap(display, RootWindow(display, screen), vi->visual, AllocNone);

	attr.border_pixel = 0;
	attr.override_redirect = fullscreen;
    attr.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask |
		PointerMotionMask | StructureNotifyMask;

	window = XCreateWindow(display, RootWindow(display, vi->screen),
			0, 0, width, height, 0, vi->depth, InputOutput, vi->visual,
			CWBorderPixel | CWColormap | CWEventMask | CWOverrideRedirect, &attr);

	if (!fullscreen){
	    Atom wmDelete;
        wmDelete = XInternAtom(display, "WM_DELETE_WINDOW", True);
        XSetWMProtocols(display, window, &wmDelete, 1);
		char *title = "OpenGL";
        XSetStandardProperties(display, window, title, title, None, NULL, 0, NULL);
	}
    XMapRaised(display, window);

	// Create a blank cursor for cursor hiding
	XColor dummy;
	char data = 0;
	Pixmap blank = XCreateBitmapFromData(display, window, &data, 1, 1);
	blankCursor = XCreatePixmapCursor(display, blank, blank, &dummy, &dummy, 0, 0);
	XFreePixmap(display, blank);


	XGrabKeyboard(display, window, True, GrabModeAsync, GrabModeAsync, CurrentTime);


	glXMakeCurrent(display, window, glContext);

	initExtensions(display);

	if (antiAliasSamples > 0){
		glEnable(GL_MULTISAMPLE_ARB);
	}

	if (fullscreen) captureMouse(!configDialog->isVisible());

	renderer = new OpenGLRenderer(window, glContext, display, screen);
	renderer->setViewport(width, height);

	antiAlias->selectItem(antiAliasSamples / 2);

	linearClamp = renderer->addSamplerState(LINEAR, CLAMP, CLAMP, CLAMP);
	defaultFont = renderer->addFont("../Textures/Fonts/Future.dds", "../Textures/Fonts/Future.font", linearClamp);
	blendSrcAlpha = renderer->addBlendState(SRC_ALPHA, ONE_MINUS_SRC_ALPHA);
	noDepthTest  = renderer->addDepthState(false, false);
	noDepthWrite = renderer->addDepthState(true,  false);
	cullNone  = renderer->addRasterizerState(CULL_NONE);
	cullBack  = renderer->addRasterizerState(CULL_BACK);
	cullFront = renderer->addRasterizerState(CULL_FRONT);

	return true;
}

void OpenGLApp::exitAPI(){
	delete renderer;

    glXMakeCurrent(display, None, NULL);
    glXDestroyContext(display, glContext);

	if (fullscreen){
		if (XF86VidModeSwitchToMode(display, screen, dmodes[0])){
			XF86VidModeSetViewPort(display, screen, 0, 0);
		}
	}
    XFree(dmodes);
	XFreeCursor(display, blankCursor);

	XDestroyWindow(display, window);

	XSync(display, False);
}

#elif defined(__APPLE__)

Boolean GetDictionaryBoolean(CFDictionaryRef theDict, const void *key){
	Boolean value = false;
	CFBooleanRef boolRef = (CFBooleanRef) CFDictionaryGetValue(theDict, key);
	if (boolRef != NULL) value = CFBooleanGetValue(boolRef);

	return value;
}

long GetDictionaryLong(CFDictionaryRef theDict, const void *key){
	long value = 0;
	CFNumberRef numRef = (CFNumberRef) CFDictionaryGetValue(theDict, key);
	if (numRef != NULL)	CFNumberGetValue(numRef, kCFNumberLongType, &value);

	return value;
}

bool OpenGLApp::initAPI(){
	initialMode = CGDisplayCurrentMode(kCGDirectMainDisplay);

	dmodes = CGDisplayAvailableModes(kCGDirectMainDisplay);
	int count = CFArrayGetCount(dmodes);

	Array <DispRes> modes;
	int foundMode = -1;
	for (int i = 0; i < count; i++){
		CFDictionaryRef mode = (CFDictionaryRef) CFArrayGetValueAtIndex(dmodes, i);

		long bitsPerPixel = GetDictionaryLong(mode, kCGDisplayBitsPerPixel);
		Boolean safeForHardware = GetDictionaryBoolean(mode, kCGDisplayModeIsSafeForHardware);
		Boolean stretched = GetDictionaryBoolean(mode, kCGDisplayModeIsStretched);

		if (bitsPerPixel < colorBits || !safeForHardware || stretched) continue;

		long width  = GetDictionaryLong(mode, kCGDisplayWidth);
		long height = GetDictionaryLong(mode, kCGDisplayHeight);
		long refreshRate = GetDictionaryLong(mode, kCGDisplayRefreshRate);

//		printf("Mode: %dx%dx%d @ %d\n", width, height, bitsPerPixel, refreshRate);

		if (width >= 640 && height >= 480){
			modes.add(newRes(width, height, i));

			if (width == fullscreenWidth && height == fullscreenHeight){
				foundMode = i;
			}
		}
	}

	resolution->clear();
	modes.sort(dComp);
	char str[64];
	for (uint i = 0; i < modes.getCount(); i++){
		sprintf(str, "%dx%d", modes[i].w, modes[i].h);
		int index = resolution->addItemUnique(str);
		if (modes[i].index == foundMode) resolution->selectItem(index);
	}

	if (fullscreen){
		if (foundMode < 0 || CGDisplaySwitchToMode(kCGDirectMainDisplay, (CFDictionaryRef) CFArrayGetValueAtIndex(dmodes, foundMode)) != kCGErrorSuccess){
			sprintf(str, "Couldn't set fullscreen to %dx%d.", fullscreenWidth, fullscreenHeight);
			ErrorMsg(str);
			fullscreen = false;
		}
	}


	Rect rect;
	if (fullscreen){
		rect.left = 0;
		rect.top  = 0;
	} else {
		long w = GetDictionaryLong(initialMode, kCGDisplayWidth);
		long h = GetDictionaryLong(initialMode, kCGDisplayHeight);

		rect.left = (w - width) / 2;
		rect.top  = (h - height) / 2;
	}
	rect.right = rect.left + width;
	rect.bottom = rect.top + height;

	WindowAttributes attributes = fullscreen? (kWindowNoTitleBarAttribute | kWindowNoShadowAttribute) : (kWindowStandardDocumentAttributes | kWindowStandardHandlerAttribute);

	OSStatus error = CreateNewWindow(kDocumentWindowClass, attributes, &rect, &window);
	if (error != noErr || window == NULL){
		ErrorMsg("Couldn't create window");
		return false;
	}

    GDHandle screen = GetGWorldDevice(GetWindowPort(window));
    if (screen == NULL){
        ErrorMsg("Couldn't get device");
        ReleaseWindow(window);
        return false;
    }

	AGLPixelFormat pixelFormat;
	while (true){
		GLint attributes[] = {
			fullscreen? AGL_FULLSCREEN : AGL_WINDOW,
			AGL_RGBA,
			AGL_DOUBLEBUFFER,
			AGL_RED_SIZE,            8,
			AGL_GREEN_SIZE,          8,
			AGL_BLUE_SIZE,           8,
			AGL_ALPHA_SIZE,         (colorBits > 24)? 8 : 0,
			AGL_DEPTH_SIZE,          depthBits,
			AGL_STENCIL_SIZE,        stencilBits,
			AGL_SAMPLE_BUFFERS_ARB, (antiAliasSamples > 0),
			AGL_SAMPLES_ARB,         antiAliasSamples,
			AGL_NONE
		};

		pixelFormat = aglChoosePixelFormat(&screen, 1, attributes);
		if (pixelFormat != NULL) break;

		antiAliasSamples -= 2;
		if (antiAliasSamples < 0){
			ErrorMsg("No suitable pixel format");
			ReleaseWindow(window);
			return false;
		}
	}

	glContext = aglCreateContext(pixelFormat, NULL);
    aglDestroyPixelFormat(pixelFormat);

	if (glContext == NULL){
		ErrorMsg("Couldn't create context");
		ReleaseWindow(window);
		return false;
	}

	if (fullscreen){
		CGCaptureAllDisplays();
		aglSetFullScreen(glContext, 0, 0, 0, 0);
	} else {
		if (!aglSetDrawable(glContext, GetWindowPort(window))){
			ErrorMsg("Couldn't set drawable");
			aglDestroyContext(glContext);
			ReleaseWindow(window);
			return false;
		}
	}

	if (!aglSetCurrentContext(glContext)){
		ErrorMsg("Couldn't make context current");
		aglDestroyContext(glContext);
		ReleaseWindow(window);
		return false;
	}

	setWindowTitle(getTitle());
    ShowWindow(window);

	initExtensions();

	if (antiAliasSamples > 0){
		glEnable(GL_MULTISAMPLE_ARB);
	}

	if (fullscreen) captureMouse(!configDialog->isVisible());

	renderer = new OpenGLRenderer(glContext);
	renderer->setViewport(width, height);

	antiAlias->selectItem(antiAliasSamples / 2);

	linearClamp = renderer->addSamplerState(LINEAR, CLAMP, CLAMP, CLAMP);
	defaultFont = renderer->addFont("../Textures/Fonts/Future.dds", "../Textures/Fonts/Future.font", linearClamp);
	blendSrcAlpha = renderer->addBlendState(SRC_ALPHA, ONE_MINUS_SRC_ALPHA);
	noDepthTest  = renderer->addDepthState(false, false);
	noDepthWrite = renderer->addDepthState(true,  false);
	cullNone  = renderer->addRasterizerState(CULL_NONE);
	cullBack  = renderer->addRasterizerState(CULL_BACK);
	cullFront = renderer->addRasterizerState(CULL_FRONT);

	return true;
}

void OpenGLApp::exitAPI(){
	delete renderer;
	
	aglSetCurrentContext(NULL);
	aglSetDrawable(glContext, NULL);
	aglDestroyContext(glContext);
	ReleaseWindow(window);

	if (fullscreen){
		CGReleaseAllDisplays();
		CGDisplaySwitchToMode(kCGDirectMainDisplay, initialMode);
	}
}

#endif
#endif

void OpenGLApp::beginFrame(){
	glColor3f(1, 1, 1);
	renderer->setViewport(width, height);
}

void OpenGLApp::endFrame(){
#if defined(_WIN32)
	if (WGL_EXT_swap_control_supported){
		wglSwapIntervalEXT(vSync? 1 : 0);
	}

	SwapBuffers(hdc);
#elif defined(LINUX)
	glXSwapBuffers(display, window);
	glFinish();
#elif defined(__APPLE__)
	aglSwapBuffers(glContext);
	glFinish();
#endif

#ifdef DEBUG
	checkForErrors();
#endif
}

bool OpenGLApp::checkForErrors(){
	int error = glGetError();
	if (error != GL_NO_ERROR){
		if (error        ==    GL_INVALID_ENUM){
			outputDebugString("GL_INVALID_ENUM");
		} else if (error ==    GL_INVALID_VALUE){
			outputDebugString("GL_INVALID_VALUE");
		} else if (error ==    GL_INVALID_OPERATION){
			outputDebugString("GL_INVALID_OPERATION");
		} else if (error ==    GL_STACK_OVERFLOW){
			outputDebugString("GL_STACK_OVERFLOW");
		} else if (error ==    GL_STACK_UNDERFLOW){
			outputDebugString("GL_STACK_UNDERFLOW");
		} else if (error ==    GL_OUT_OF_MEMORY){
			outputDebugString("GL_OUT_OF_MEMORY");
		} else if (error ==    GL_INVALID_FRAMEBUFFER_OPERATION_EXT){
			outputDebugString("GL_INVALID_FRAMEBUFFER_OPERATION_EXT");
		} else {
			outputDebugString("Unrecognized OpenGL error");
		}

		return true;
	}

	return false;
}

void OpenGLApp::onSize(const int w, const int h){
	BaseApp::onSize(w, h);

	if (glContext != NULL){
#if defined(__APPLE__)
		aglUpdateContext(glContext);
#endif
		glViewport(0, 0, w, h);
	}
}

bool OpenGLApp::captureScreenshot(Image &img){
	ubyte *pixels = img.create(FORMAT_RGB8, width, height, 1, 1);
	ubyte *flipped = new ubyte[width * height * 3];

	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, flipped);
	for (int y = 0; y < height; y++){
		memcpy(pixels + y * width * 3, flipped + (height - y - 1) * width * 3, width * 3);
	}
	delete [] flipped;

	return true;
}
