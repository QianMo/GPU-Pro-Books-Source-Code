//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//  Scene Graph 3D                                                          //
//  Georgios Papaioannou, 2009                                              //
//                                                                          //
//  This is a free, extensible scene graph management library that works    //
//  along with the EaZD deferred renderer. Both libraries and their source  //
//  code are free. If you use this code as is or any part of it in any kind //
//  of project or product, please acknowledge the source and its author.    //
//                                                                          //
//  For manuals, help and instructions, please visit:                       //
//  http://graphics.cs.aueb.gr/graphics/                                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

#ifndef _INPUTDEVICES3D_
#define _INPUTDEVICES3D_

#ifdef WIN32
#define DIRECTINPUT_VERSION 0x0800
#include <Dinput.h>
#else
#include <sstream>

#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <X11/cursorfont.h>
#endif

class DeviceAxis3D
{
public:
	float range_min, range_max, value;
	float getNormalizedValue() {return 2*(value-range_min)/(range_max-range_min)-1;}
};

class InputDevice3D
{
protected:
	DeviceAxis3D axes[16];
	bool buttons[32];
	int num_buttons;
	int num_axes;
	int err_code;
	bool initialized;
public:
	InputDevice3D();
	virtual ~InputDevice3D();
	virtual void init();
	virtual void reset();
	virtual void update(double dt)=0;
	virtual void passData(void *);
	void setRanges(int axis, float rmin, float rmax);
	float getNormalizedValue(int axis);
	float getValue(int axis);
	bool getButton(int i);
	int getButtons();
	int getAxes();
};

class MouseKeyboardDevice3D: public InputDevice3D
{
protected:
#ifdef WIN32
	LPDIRECTINPUTDEVICE8 pKeyboard;
	LPDIRECTINPUTDEVICE8 pMouse;
	LPDIRECTINPUT8		 pDI;
	HWND				 hWnd;
	HINSTANCE			 hInstance;
#else
    std::string          hostName;
    int                  displayNum;
    int                  screenNum;
    Display *            pDisplay;
    Window               pWindow;        // the GLUT application window
    Window               eventWindow;    // the InputOnly window responsible for the events

    void setHostnameDisplayNumScreenNum (const std::string& displayName);
    std::string displayName() const;
#endif
	int mouse_x, mouse_y, prev_x, prev_y;
	char raw_key_map[256];
public:
	MouseKeyboardDevice3D();
	~MouseKeyboardDevice3D();
	virtual void update(double dt);
	virtual void init();
};

class JoystickDevice3D: public InputDevice3D
{
protected:
	bool enabled;
#ifdef WIN32
	// windows implementation via DirectX
	HWND				 hWnd;
	HINSTANCE			 hInstance;
	DIDEVCAPS            diDevCaps;
	DIJOYSTATE2          js;
#else
    std::string          hostName;
    int                  displayNum;
    int                  screenNum;
    Display *            pDisplay;
    Window               pWindow;        // the GLUT application window
    Window               eventWindow;    // the InputOnly window responsible for the events

    void setHostnameDisplayNumScreenNum (const std::string& displayName);
    std::string displayName() const;
#endif

public:
	JoystickDevice3D(int dev=0); // dev represents the enum order reported by the system 0,1...
	~JoystickDevice3D();
	virtual void update(double dt);
	virtual void init();
	bool isEnabled() {return enabled;} //upon successful initialization returns true
	int  deviceID;
#ifdef WIN32
	LPDIRECTINPUT8		 pDI; // required public to be accessible by callback func.
	LPDIRECTINPUTDEVICE8 pJoystick;
	int  enum_device;
#endif
};

#endif

