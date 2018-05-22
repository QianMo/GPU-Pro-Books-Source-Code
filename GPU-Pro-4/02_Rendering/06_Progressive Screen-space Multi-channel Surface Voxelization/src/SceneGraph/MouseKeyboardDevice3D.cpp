//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//  Scene Graph 3D                                                          //
//  Georgios Papaioannou, 2009 (Direct3D support)                           //
//  Athanasios Gaitatzes, 2009 (X11 support)                                //
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

#include "SceneGraph.h"

#ifdef WIN32
#define DIRELEASE() \
{ \
	if (pDI) \
	{   if (pKeyboard) \
        { \
            pKeyboard->Unacquire(); \
            pKeyboard->Release(); \
            pKeyboard = NULL; \
        } \
		if (pMouse) \
        { \
            pMouse->Unacquire(); \
            pMouse->Release(); \
            pMouse = NULL; \
        } \
        pDI->Release(); \
        pDI = NULL; \
    } \
}
#endif

MouseKeyboardDevice3D::MouseKeyboardDevice3D()
{
#ifdef WIN32
	pDI = NULL;
	pKeyboard = NULL;
	pMouse = NULL;
#else
    pDisplay = NULL;
#endif
	num_buttons = 3;
	num_axes = 5;
	setRanges(0,-1,1);
	setRanges(1,-1,1);
	setRanges(2,-1,1);
	setRanges(3,-1,1);
	setRanges(4,-1,1);
}

#ifndef WIN32
void
MouseKeyboardDevice3D::setHostnameDisplayNumScreenNum (const std::string& displayName)
{
    std::string::size_type colon = displayName.find_last_of (':');
    std::string::size_type point = displayName.find_last_of ('.');

    if (point != std::string::npos &&
        colon == std::string::npos &&
        point < colon)
        point = std::string::npos;

    hostName = (colon == std::string::npos) ? "" : displayName.substr (0, colon);

    std::string::size_type startOfDisplayNum = (colon == std::string::npos) ? 0 : colon + 1;
    std::string::size_type endOfDisplayNum = (point == std::string::npos) ?  displayName.size () : point;

    displayNum = 0;
    if (startOfDisplayNum < endOfDisplayNum)
        displayNum = atoi (displayName.substr (startOfDisplayNum, endOfDisplayNum-startOfDisplayNum).c_str ());

    screenNum = 0;
    if (point != std::string::npos && point+1 < displayName.size ())
        screenNum = atoi (displayName.substr (point+1, displayName.size ()-point-1).c_str ());
}

std::string
MouseKeyboardDevice3D::displayName () const
{
    std::stringstream ostr;
    ostr << hostName << ':' << displayNum << '.' << screenNum;
    return ostr.str ();
}

#define MAX_PROPERTY_VALUE_LEN 4096

/* Lifted from wmctrl */
static char *
getXInternAtom (Display * disp, Window win,
                Atom xa_prop_type, char * prop_name, unsigned long *size)
{
    Atom    xa_prop_name;
    Atom    xa_ret_type;
    int     ret_format;
    unsigned long ret_nitems;
    unsigned long ret_bytes_after;
    unsigned long tmp_size;
    unsigned char *ret_prop;
    char  *ret;

    xa_prop_name = XInternAtom (disp, prop_name, False);

    /*
    // MAX_PROPERTY_VALUE_LEN / 4 explanation (XGetWindowProperty manpage):
    // long_length = Specifies the length in 32-bit multiples of the data to be
    // retrieved. NOTE: see
    // http://mail.gnome.org/archives/wm-spec-list/2003-March/msg00067.html In
    // particular: When the X window system was ported to 64-bit
    // architectures, a rather peculiar design decision was made. 32-bit
    // quantities such as Window IDs, atoms, etc, were kept as longs in the
    // client side APIs, even when long was changed to 64 bits.
    */
    if (XGetWindowProperty (disp, win, xa_prop_name, 0,
        MAX_PROPERTY_VALUE_LEN / 4, False, xa_prop_type,
        &xa_ret_type, &ret_format, &ret_nitems, &ret_bytes_after,
        &ret_prop) != Success)
        return NULL;

    if (xa_ret_type != xa_prop_type)
    {
        XFree (ret_prop);
        return NULL;
    }

    /* null terminate the result to make string handling easier */
    tmp_size = (ret_format / 8) * ret_nitems;
    /* Correct 64 Architecture implementation of 32 bit data */
    if (ret_format == 32)
        tmp_size *= sizeof (long) / 4;
    ret = (char *) malloc (tmp_size + 1);
    memcpy (ret, ret_prop, tmp_size);
    ret[tmp_size] = '\0';

    if (size)
        *size = tmp_size;

    XFree (ret_prop);
    return ret;
}

/* Lifted from wmctrl */
static Window *
getXWindows (Display * disp, unsigned long *size)
{
    Window *wins_list, window = DefaultRootWindow (disp);

    if ((wins_list = (Window *) getXInternAtom (disp, window,
                                XA_WINDOW, (char *) "_NET_CLIENT_LIST", size)) == NULL)
    {
        if ((wins_list = (Window *) getXInternAtom (disp, window,
                                    XA_CARDINAL, (char *) "_WIN_CLIENT_LIST", size)) == NULL)
        {
            EAZD_TRACE ("MouseKeyboardDevice3D getXWindows() : ERROR - "
                        "Cannot get client list properties _NET_CLIENT_LIST or _WIN_CLIENT_LIST.");
            return NULL;
        }
    }

    return wins_list;
}

static Window
FindWindow (Display *disp, char * className, char * windowName)
{
    // get us the window
    Window *possibleWindows;
    unsigned long numWindows = 0, i;

    possibleWindows = getXWindows (disp, &numWindows);
    numWindows /= sizeof (Window);

    for (i = 0; i < numWindows; i++)
    {
        if (STR_EQUAL (getXInternAtom (disp, possibleWindows[i],
                                       XA_STRING, (char *) "WM_NAME", NULL),
                       windowName))
            break;
    }

    if (i < numWindows)
        return possibleWindows[i];
    else
        return 0;
}
#endif

void MouseKeyboardDevice3D::init()
{
	if (initialized)
		return;

#ifdef WIN32
	HRESULT hr;
	//char consoleTitle[512];
	
	// init direct input interface
	//GetConsoleTitleA(consoleTitle, 512);
	//hWnd = FindWindowA(NULL, consoleTitle);
    hWnd = FindWindowA("GLUT", NULL);
	hInstance = (HINSTANCE)GetWindowLong(hWnd, GWL_HINSTANCE);
	hr = DirectInput8Create(hInstance, DIRECTINPUT_VERSION, IID_IDirectInput8, (void**)&pDI, NULL);
	if FAILED(hr)
	{
		err_code = SCENE_GRAPH_ERROR_DIRECTINPUT;
		DIRELEASE();
		return;
	}
	//init keyboard input device
	pDI->CreateDevice(GUID_SysKeyboard, &pKeyboard, NULL);
	hr = pKeyboard->SetDataFormat(&c_dfDIKeyboard);
	hr = pKeyboard->SetCooperativeLevel(hWnd, DISCL_BACKGROUND | DISCL_NONEXCLUSIVE);
	hr = pKeyboard->Acquire();
	
	if (FAILED(hr))
	{
		err_code = SCENE_GRAPH_ERROR_DEVICE_INIT;
		DIRELEASE();
		return;
	}

	//init mouse input device
	hr = pDI->CreateDevice(GUID_SysMouse, &pMouse, NULL);
	if (FAILED(hr))
	{
		err_code = SCENE_GRAPH_ERROR_DEVICE_INIT;
		DIRELEASE();
		return;
	}
	hr = pMouse->SetDataFormat(&c_dfDIMouse);
	if (FAILED(hr))
	{
		err_code = SCENE_GRAPH_ERROR_DEVICE_INIT;
		DIRELEASE();
		return;
	}
	hr = pMouse->SetCooperativeLevel(hWnd, DISCL_NONEXCLUSIVE | DISCL_BACKGROUND);
	if (FAILED(hr))
	{
		err_code = SCENE_GRAPH_ERROR_DEVICE_INIT;
		DIRELEASE();
		return;
	}
	/*
	HANDLE hMouseEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
	if (hMouseEvent == NULL)
	{
		err_code = SCENE_GRAPH_ERR_DEVICE_INIT;
		DIRELEASE();
		return;
	}
	hr = pMouse->SetEventNotification(hMouseEvent);
	if (FAILED(hr))
	{
	   	err_code = SCENE_GRAPH_ERR_DEVICE_INIT;
		DIRELEASE();
		return;
	}
	*/
	#define SAMPLE_BUFFER_SIZE  8

	DIPROPDWORD dipdw;
    dipdw.diph.dwSize       = sizeof(DIPROPDWORD);
    dipdw.diph.dwHeaderSize = sizeof(DIPROPHEADER);
    dipdw.diph.dwObj        = 0;
    dipdw.diph.dwHow        = DIPH_DEVICE;
    dipdw.dwData            = SAMPLE_BUFFER_SIZE;
	hr = pMouse->SetProperty(DIPROP_BUFFERSIZE, &dipdw.diph);
	dipdw.diph.dwSize       = sizeof(DIPROPDWORD);
    dipdw.diph.dwHeaderSize = sizeof(DIPROPHEADER);
    dipdw.diph.dwObj        = 0;
    dipdw.diph.dwHow        = DIPH_DEVICE;
    dipdw.dwData            = DIPROPAXISMODE_REL;
	hr = pMouse->SetProperty(DIPROP_AXISMODE, &dipdw.diph);
	if (FAILED(hr))
	{
		err_code = SCENE_GRAPH_ERROR_DEVICE_INIT;
		DIRELEASE();
		return;
	}

#else
    // get us the display
    const char* ptr = 0;
    if ((ptr = getenv ("DISPLAY")) != 0)
        setHostnameDisplayNumScreenNum (ptr);

    if (! (pDisplay = XOpenDisplay (displayName ().c_str ())))
    {
        EAZD_TRACE ("MouseKeyboardDevice3D::init() : ERROR - "
                    "Unable to open display \"" << XDisplayName (displayName ().c_str ()) << "\".");
        err_code = SCENE_GRAPH_ERROR_DEVICE_INIT;
        XCloseDisplay (pDisplay);
        pDisplay = 0;

        return;
    }
    EAZD_TRACE ("Display initialized");

    if (! (pWindow = FindWindow (pDisplay, NULL, (char *) "Deferred Renderer Demo")))
    {
        EAZD_TRACE ("MouseKeyboardDevice3D::init() : ERROR - "
                    "Unable to find window on \"" << XDisplayName (displayName ().c_str ()) << "\".");
        err_code = SCENE_GRAPH_ERROR_DEVICE_INIT;
        XCloseDisplay (pDisplay);

        return;
    }
    EAZD_TRACE ("Window initialized");

    XWindowAttributes wattr;

    if (! XGetWindowAttributes (pDisplay, pWindow, &wattr))
    {
        EAZD_TRACE ("MouseKeyboardDevice3D::init() : ERROR - "
                    "Can not get window attributes.");
        err_code = SCENE_GRAPH_ERROR_DEVICE_INIT;
        XCloseDisplay (pDisplay);
        pWindow = 0;

        return;
    }

    XSetWindowAttributes swatt;
    // need to set the gravity to southeast so that when the parent window
    // resizes the eventWindow moves thus creating a GravityNotify event
    // from the StructureNotifyMask
    swatt.win_gravity = SouthEastGravity;
    swatt.cursor = XCreateFontCursor (pDisplay, XC_crosshair);
    swatt.event_mask = StructureNotifyMask | KeyPressMask |
        KeyReleaseMask | PointerMotionMask | ButtonPressMask |
        ButtonReleaseMask | KeymapStateMask;
    unsigned long swatt_mask = CWWinGravity | CWCursor | CWEventMask;

    // the border_width for an InputOnly window must be zero,
    // the depth for an InputOnly window must be zero,
    // or a BadMatch error results.
    eventWindow = XCreateWindow (pDisplay, pWindow,
        wattr.x, wattr.y, wattr.width, wattr.height,
        0, CopyFromParent, InputOnly, CopyFromParent,
        swatt_mask, &swatt);

    XMapWindow (pDisplay, eventWindow);

    XFlush (pDisplay);
    XSync (pDisplay, 0);
#endif
	mouse_x = mouse_y = prev_x = prev_y = 0;
	memset(raw_key_map,0,256);
	
	err_code = SCENE_GRAPH_ERROR_NONE;
	initialized = true;
}

MouseKeyboardDevice3D::~MouseKeyboardDevice3D()
{
#ifdef WIN32
	DIRELEASE();
#else
    XCloseDisplay (pDisplay);
    pDisplay = 0;
#endif
}

#define KEYDOWN(name, key) (name[key] & 0x80)

void MouseKeyboardDevice3D::update(double dt)
{
	if (! initialized)
		return;

#ifdef WIN32
	// read keyboard
	HRESULT hr = pKeyboard->GetDeviceState(sizeof(raw_key_map),(LPVOID)&raw_key_map);
	if (! FAILED(hr))
    {
		if (KEYDOWN(raw_key_map, DIK_RIGHT)||KEYDOWN(raw_key_map, DIK_D))
			axes[1].value = 1.0f;
		else if (KEYDOWN(raw_key_map, DIK_LEFT)||KEYDOWN(raw_key_map, DIK_A))
			axes[1].value = -1.0f;
		else
			axes[1].value = 0.0f;
		if (KEYDOWN(raw_key_map, DIK_UP)||KEYDOWN(raw_key_map, DIK_W))
			axes[0].value = 1.0f;
		else if (KEYDOWN(raw_key_map, DIK_DOWN)||KEYDOWN(raw_key_map, DIK_S))
			axes[0].value = -1.0f;
		else
			axes[0].value = 0.0f;
	}
    //read mouse
	DIDEVICEOBJECTDATA od[8];
    DWORD dwElements = 8;
	hr = pMouse->Acquire();
	hr = pMouse->Poll();
	
	hr = pMouse->GetDeviceData(sizeof(DIDEVICEOBJECTDATA),
                                     od, &dwElements, 0);
	
	axes[2].value = axes[3].value = axes[4].value = 0.0f;
	//buttons[0]=buttons[1]=buttons[2]=false;
	if (hr != DIERR_INPUTLOST && ! FAILED(hr) && dwElements > 0)
    {
		for (DWORD i=0; i<dwElements; i++)
		{
			switch (od[i].dwOfs)
			{
			case DIMOFS_X:
				axes[2].value+=4.0f*(long)(od[i].dwData)/(float)GetSystemMetrics(SM_CXSCREEN);
				break;
			case DIMOFS_Y:
					axes[3].value+=4.0f*(long)(od[i].dwData)/(float)GetSystemMetrics(SM_CYSCREEN);
				break;
			//case DIMOFS_Z:
			//	axes[4].value+=(long)(od[i].dwData);
				break;
			case DIMOFS_BUTTON0:
				buttons[0] = (od[i].dwData & 0x80) ? true : false;
				break;
			case DIMOFS_BUTTON1:
				buttons[2] = (od[i].dwData & 0x80) ? true : false;
				break;
			case DIMOFS_BUTTON2:
				buttons[1] = (od[i].dwData & 0x80) ? true : false;
				break;
			}
		}
	}
	// EAZD_TRACE ("mouse(x,y,z) = (" << axes[2].value << ',' << axes[3].value << ',' << axes[4].value << ')');
#else
    while (XPending (pDisplay))
    {
        XEvent event;
        XNextEvent (pDisplay, &event);

        switch (event.type)
        {
        // The X server doesn't generate Expose events on InputOnly windows
        case Expose:
            break;
        case GravityNotify:
        {
            // remove redundant GravityNotify events
            while (XEventsQueued (pDisplay, QueuedAfterReading) > 0)
            {
                XEvent aheadEvent;
                XPeekEvent (pDisplay, &aheadEvent);

                if (aheadEvent.type != GravityNotify ||
                    aheadEvent.xgravity.window != event.xgravity.window)
                    break;
                else
                    XNextEvent (pDisplay, &event);
            }

            if (event.xgravity.window == eventWindow)
            {
                EAZD_TRACE ("Gravity Notify event received");

                XWindowAttributes watt;
                XGetWindowAttributes (pDisplay, pWindow, &watt );
                XMoveResizeWindow (pDisplay, eventWindow, watt.x, watt.y,
                                   watt.width, watt.height);

                XFlush (pDisplay);
                XSync (pDisplay, 0);
            }
        }
            break;
        case ReparentNotify:
            EAZD_TRACE ("Reparent Notify event received");
            break;
        case DestroyNotify:
            EAZD_TRACE ("Destroy Notify event received");
            break;

        case MappingNotify:
            EAZD_TRACE ("Key mapping changed");
            XRefreshKeyboardMapping (&event.xmapping);
            break;
        case KeyPress:
        {
            KeySym keysym;
            keysym = XLookupKeysym ((XKeyEvent *) &event, 0);
#ifdef EAZD_DEBUG
            char c;
            XLookupString (&event.xkey, &c, 1, &keysym, 0);
#endif
         // EAZD_TRACE ("Key pressed: " << c << " " << std::hex << "0x" << keysym << std::dec);

                                   // include the keypad
            if (keysym == XK_Right || keysym == XK_KP_Right ||
                keysym == XK_D || keysym == XK_d)
                axes[1].value = 1.0f;
            else
            if (keysym == XK_Left || keysym == XK_KP_Left ||
                keysym == XK_A || keysym == XK_a)
                axes[1].value = -1.0f;
            else
            if (keysym == XK_Up || keysym == XK_KP_Up ||
                keysym == XK_W || keysym == XK_w)
                axes[0].value = 1.0f;
            else
            if (keysym == XK_Down || keysym == XK_KP_Down ||
                keysym == XK_S || keysym == XK_s)
                axes[0].value = -1.0f;
            else
            {
                // need to send the unwanted events to the parent
                event.xkey.window = pWindow;
                event.xkey.send_event = true;

                XSendEvent (event.xkey.display, event.xkey.window,
                            true, KeyPressMask, &event);
                EAZD_TRACE ("Sending to parent ...");
            }
        }
            break;
        case KeyRelease:
        {
            KeySym keysym;
            keysym = XLookupKeysym ((XKeyEvent *) &event, 0);
#ifdef EAZD_DEBUG
            char c;
            XLookupString (&event.xkey, &c, 1, &keysym, 0);
#endif
         // EAZD_TRACE ("Key released: " << c << " " << std::hex << "0x" << keysym << std::dec);

            if (keysym == XK_Right || keysym == XK_KP_Right ||
                keysym == XK_D || keysym == XK_d ||
                keysym == XK_Left || keysym == XK_KP_Left ||
                keysym == XK_A || keysym == XK_a ||
                keysym == XK_Up || keysym == XK_KP_Up ||
                keysym == XK_W || keysym == XK_w ||
                keysym == XK_Down || keysym == XK_KP_Down ||
                keysym == XK_S || keysym == XK_s)
            {
                axes[1].value = 0.0f;
                axes[0].value = 0.0f;
            }
            else
            {
                // need to send the unwanted events to the parent
                event.xkey.window = pWindow;
                event.xkey.send_event = true;

                XSendEvent (event.xkey.display, event.xkey.window,
                            true, KeyReleaseMask, &event);
                EAZD_TRACE ("Sending to parent ...");
            }
        }
            break;
        case ButtonPress:
        {
            unsigned int which = event.xbutton.button - 1;
            int x = event.xbutton.x;
            int y = event.xbutton.y;
            EAZD_TRACE ("Button pressed: " << which << " (" << x << ',' << y << ')');
            buttons[which] = true;
        }
            break;
        case ButtonRelease:
        {
            unsigned int which = event.xbutton.button - 1;
            int x = event.xbutton.x;
            int y = event.xbutton.y;
            EAZD_TRACE ("Button released: " << which << " (" << x << ',' << y << ')');
            buttons[which] = false;
        }
            break;
        case MotionNotify:
        {
            static int old_x, old_y;

            int x = event.xmotion.x;
            int y = event.xmotion.y;
         // EAZD_TRACE ("Motion Notify: old(x,y) = (" << old_x << ',' << old_y << ") (x,y) = (" << x << ',' << y << ')');

			axes[2].value=0; // x-old_x;
			axes[3].value=0; // y-old_y;
			axes[4].value=0;

            old_x = x;
            old_y = y;
        }
            break;
        }
    }
#endif
}

#undef KEYDOWN
#undef DIRELEASE

