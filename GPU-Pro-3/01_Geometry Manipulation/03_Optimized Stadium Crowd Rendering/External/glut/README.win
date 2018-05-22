

			GLUT for Win32 README
			---------------------


VERSION/INFO:

	This is GLUT for Win32 version 3.6 as of December 10th 1997.
	See the COPYRIGHT section for distribution and copyright notices.
	Send all bug reports/questions to Nate Robins (ndr@pobox.com)
	and Mark Kilgard (mjk@sgi.com).

	For information about GLUT for Win32, see the web page:
	www.pobox.com/~ndr/glut.html or subscribe to the mailing list:
        mail to majordomo@perp.com with "subscribe glut-win32" in the
        body of the message.

	For general information about GLUT, see the GLUT web page:
	http://reality.sgi.com/mjk/glut3/glut3.html

	There are two versions of the library.  One for use with the
        Microsoft implementation of OpenGL (opengl32) and one for use
        with the SGI implementation of OpenGL (opengl).  The trailing
        '32' indicates a Microsoft implementation.  Therefore, if you
        are using opengl32.dll, use glut32.dll and if you are using
        opengl.dll use glut.dll.


COMPILING/INSTALLATION:

	o  Precompiled versions of the DLL and import library can be
	   found on the GLUT for Win32 web page mentioned above.

	o  Edit the glutwin32.mak file to reflect your system configuration
	   (see the glutwin32.mak file for more information).

	o  Type "glutmake" to make everything.  Try "glutmake clean" to
           delete all intermediate files, and "glutmake clobber" to
           delete all intermediate files and executables.  The build
	   system will automatically install everything in the places
	   specified in the glutwin32.mak file.

	o  Other ways to compile GLUT for Windows:

	    For no debugging info (ie, glutmake):   nmake nodebug=1
	    For debugging info:                     nmake
	    For working set tuner info:             nmake tune=1
	    For call attribute profiling info:      nmake profile=1


BORLAND NOTES:

	From what I understand, Borland supplies a utility that
        converts Microsoft Visual C++ .libs into Borland compatible
        files.  Therefore, the best method for Borland users is
        probably to get the precompiled versions of the library and
        convert the library.


MISC NOTES:

	o  Overlay support is currently not implemented.  I _had_ it
           implemented, but Microsoft has such LAME fluidity in its
           header files from version to version that I've decided it
           is easier just to wait for everything to settle down and
           (hopefully) vendors to provide better support for overlays.

	o  To customize the windows icon, you can use the resource name
	   GLUT_ICON.  For example, create an icon named "glut.ico", and
	   create a file called glut.rc that contains the following:
	   GLUT_ICON ICON glut.ico
	   then compile the glut.rc file with the following:
	   rc /r glut
           and link the resulting glut.res file into your executable
           (just like you would an object file).



IMPLEMENTATION DEPENDENT DIFFERENCES:

	There are a few differences between the Win32 version of GLUT
        and the X11 version of GLUT.  Those are outlined here.  Note
        that MOST of these differences are allowed by the GLUT
        specification.  Bugs and unsupported features are outlined in
        the UNSUPPORTED/BUGS section.

	o  glutInit: 
           The following command line options have no meaning (and are
           ignored) in GLUT for Win32:
	   -display, -indirect, -direct, -sync.

	o  glutInitWindowPosition, glutPositionWindow:
	   Win32 has two different coordinate systems for windows.
           One is in terms of client space and the other is the whole
           window space (including the decorations).  If you
           glutPositionWindow(0, 0), GLUT for Win32 will place the
           window CLIENT area at 0, 0.  This will cause the window
           decorations (title bar and left edge) to be OFF-SCREEN, but
           it gives the user the most flexibility in positioning.
           HOWEVER, if the user specifies glutInitWindowPosition(0, 0),
           the window is placed relative to window space at 0, 0.
           This will cause the window to be opened in the upper left
           corner with all the decorations showing.  This behaviour is
           acceptable under the current GLUT specification.

	o  glutSetIconTitle, glutSetWindowTitle: 
	   There is no separation between Icon title and Window title
           in Win32.  Therefore, changing the icon title will change
           the window title and vice-versa.  This could be worked around
	   by saving the icon and window titles and changing the window
           title when the state of the window is changed.

	o  glutSetCursor:
	   As indicated in the GLUT specification, cursors may be
           different on different platforms.  This is the case in GLUT
           for Win32.  For the most part, the cursors will match the
           meaning, but not necessarily the shape.  Notable exceptions
           are the GLUT_CURSOR_INFO & GLUT_CURSOR_SPRAY which use the
           crosshair cursor and the GLUT_CURSOR_CYCLE which uses the
           'no' or 'destroy' cursor in Win32.

	o  glutVisibilityFunc: 
	   Win32 seems to be unable to determine if a window is fully
           obscured.  Therefore, the visibility of a GLUT window is
           only reflected by its Iconic, Hidden or Shown state.  That
           is, even if a window is fully obscured, in GLUT for Win32,
           it is still "visible".

	o  glutEntryFunc: 
           Window Focus is handled differently in Win32 and X.
           Specifically, the "window manager" in Win32 uses a "click to
           focus" policy.  That is, in order for a window to receive
           focus, a mouse button must be clicked in it.  Likewise, in
           order for a window to loose focus, a mouse button must be
           clicked outside the window (or in another window).
           Therefore, the Enter and Leave notification provided by GLUT
           may behave differently in the Win32 and in X11 versions.
           There is a viable workaround for this.  A program called
           "Tweak UI" is provided by Microsoft which can be used to
           change the focus policy in Win32 to "focus follows mouse".
           It is available from the Microsoft Web Pages:
	   http://www.microsoft.com/windows/software/PowerToy.htm

	o  glutCopyColormap:
	   GLUT for Win32 always copies the colormap.  There is never
           any sharing of colormaps.  This is probably okay, since
	   Win32 merges the logical palette and the physical palette
	   anyway, so even if there are two windows with totally
	   different colors in their colormaps, Win32 will find a
	   (hopefully) good match between them.

	o  glutIdleFunc + menus:
	   The glut idle function will NOT be called when a menu is
           active.  This causes all animation to stop when a menu is
           active (in general, this is probably okay).  Timer
           functions will still fire, however.  If the timer callback
           draws into the rendering context, the drawing will not show
           up until after the menu has finished, though.


UNSUPPORTED/BUGS:

	o  glutAttachMenu:
	   Win32 only likes to work with left and right mouse buttons.
           Especially so with popup menus.  Therefore, when attaching
           the menu to the middle mouse button, the LEFT mouse button
           must be used to select from the menu (c'mon Microsoft -
	   can't you do more than 2 buttons?).

	o  glutSpaceball*, glutButtonBox*, glutTablet*, glutDials*:
	   None of the special input devices are supported at this
           time.

	o  When resizing or moving a GLUT for Win32 window, no updating
           is performed.  This causes the window to leave "tracks" on
           the screen when getting bigger or when previously obscured
           parts are being revealed.  I put in a bit of a kludgy
           workaround for those that absolutely can't have the weird
           lines.  The reshape callback is called multiple times for
           reshapes.  Therefore, in the reshape callback, some drawing
           can be done.

	o  The video resizing capabilities of GLUT 3.3+ for X11 is
           currently unimplemented (this is probably ok, since it
           really isn't part of the spec until 4.0).  I doubt that
           this will ever be part of GLUT for Win32, since there is no
           hardware to support it.  A hack could simply change the
           resolution of the desktop.


CHANGES/FIXES:

	(May 22, '97)
	o  Menus don't work under Windows 95
	x  Fixed!  Added a unique identifier to each menu item, and a 
           search function to grab a menu item given the unique identifier.

	(May 21, '97)
	o  A few minor bug fixes here and there.
	x  Thanks to Bruce Silberman and Chris Vale for their help with
           this.  We now have a DLL!

	(Apr 25, '97)
	o  DLL version of the library is coming (as soon as I figure out
	   how to do it -- if you know, let me know).
	x  Thanks to Bruce Silberman and Chris Vale for their help with
           this.  We now have a DLL!

	(Apr 24, '97)
	x  Added returns to KEY_DOWN etc messages so that the F10 key
           doesn't toggle the system menu anymore.

	(Apr 7, '97)
	o  Palette is incorrect for modes other than TrueColor.
	x  Fixed this by forcing a default palette in modes that aren't
	   Truecolor in order to 'simulate' it.  The applications
           program shouldn't have to do this IMHO, but I guess we
           can't argue with Microsoft (well, we can, but what good
           will it do?).

	(Apr 2, '97)
	x  Added glut.ide file for Borland users.

	(Apr 2, '97)
	x  Fixed a bug in the WM_QUERYNEWPALETTE message.  Wasn't
           checking for a null colormap, then de-ref'd it.  Oops.

	(Mar 13, '97)
	o  glutTimerFunc: 
	   Currently, GLUT for Win32 programs busy waits when there is
           an outstanding timer event (i.e., there is no select()
           call).  I haven't found this to be a problem, but I plan to
           fix it just because I can't bear the thought of a busy wait.
	x  Added a timer event and a wait in the main loop.  This fixes
           the CPU spike.

	(Mar 11, '97)
	x  Fixed subwindow visibility.  The visibility callback of
	   subwindows wasn't being called, now it is.

	(Mar 11, '97)
	o  glutGetHDC, glutGetHWND:
	   In order to support additional dialog boxes, wgl fonts, and
	   a host of other Win32 dependent structures, two functions
           have been added that operate on the current window in GLUT.
           The first (glutGetHDC) returns a handle to the current
           windows device context.  The second (glutGetHWND) returns
           handle to the current window.
	x  Took these out to preserve GLUT portability.

	(Mar 11, '97)
	x  Fixed the glutWarpPointer() coordinates.  Were relative to
           the screen, now relative to window (client area) origin
           (which is what they're supposed to be).

	(Mar 11, '97)
	o  glutCreateMenu, glutIdleFunc:
	   Menu's are modal in Win32.  That is, they don't allow any
	   messages to be processed while they are up.  Therefore, if
	   an idle function exists, it will not be called while
	   processing a menu.
	x  Fixed!  I've put in a timer function that fires every
           millisecond while a menu is up.  The timer function handles
           idle and timer events only (which should be the only
           functions that are firing when a menu is up anyway).

	(Mar 7 '97)
	x  Fixed minor bugs tracked down by the example programs.

	(Mar 6, '97)
	x  Merged 3.3 GLUT for X11 into 3.2 GLUT for Win32.  New code
           structure allows for EASY merging!

	o  In Win32, the parent gets the right to set the cursor of
           any of its children.  Therefore, a child windows cursor
           will 'blink' between its cursor and its parent.
	x  Fixed this by checking whether the cursor is in a child
           window or not.

	(Feb 28 '97)
	o  On initial bringup apps are getting 2 display callbacks.
	x  Fixed by the Fev 28 re-write.

	o  Some multiple window (not subwindow) functionality is messed up.
           See the sphere.exe program.
	x  Fixed by the Feb 28 re-write.

	o  GLUT for Win32 supports color index mode ONLY in a paletted
           display mode (i.e., 256 or 16 color mode).
	x  Fixed this in the re-write.  If you can get a color index
           visual (pixel format) you can use color index mode.

	(Feb 28 '97)
	o  Quite a few bugs (and incompatibilities) were being caused
           by the structure that I used in the previous port of GLUT.
           Therefore I decided that it would be best to "get back to
           the roots".  I re-implemented most of glut trying to stick
           with the structure layed out by Mark.  The result is a much
	   more stable version that passes ALL (!) (except overlay)
           the tests provided by Mark.  In addition, this new
           structure will allow future enhancements by Mark to be
	   integrated much more quickly into the Win32 version.  Also,
           I'm now ordering the bugs in reverse, so that the most
           recently fixed appear at the top of the list.

	(9/8/96)
	o  Changed the glutGetModifiers code to produce an error if not
	   called in the core input callbacks.

	(9/11/96)
	o  If the alt key is pressed with more than one other modifier key
	   it acts as if it is stuck -- it stays selected until pressed
	   and released again.
	x  Fixed. 

	(9/12/96)
	o  When a submenu is attached to a menu, sometimes a GPF occurs.
	   Fixed.  Needed to set the submenu before referencing it's members.

	o  Kenny: Also, one little problem, I attached the menu to the 
	   right-button, but when the left-button is pressed I detach
	   it to give the right-button new meaning; if I pop-up the menu and I
	   don't want to select anything, like most users, I click off of the
	   menu to make it disappear. When I do this, I get a GLUT error and 
	   the program terminates because I am altering the menu attachment 
	   from within the button press while the menu is active. 
	x  Fixed.  Needed to finish the menu when the user presses the button,
	   not just when a button is released.

	o GLUT for Win32 emulates a middle mouse button by checking if
           both mouse
	   buttons are down.  This causes a lot of problems with the menu
	   and other multiple mouse button things.  
	x  Fixed.  No more middle mouse button emulation.  Perhaps it would
	   be a good idea to emulate the middle mouse button (if not present)
	   with a key?

	(9/15/96)
	o  Added code to accept a user defined icon.  If no icon is provided,
	   a default icon is loaded.

	(9/19/96)
	o  Shane: Command line options seem to be screwed up. (9/13)
	x  Fixed.  The geometry command line was broken, and so was the
	   gldebug command line.

	o  Fixed a bug in the default glut reshape.  It was looking for the
	   parent of the current window and GPF'ing if there wasn't a parent.
	   Put in a check for a parent, and if none is there, use the
	   child.

	o  Idle function sucks up all processor cycles. (9/8/96)
	x  I don't know if this is avoidable.  If you have a tight rendering
	   loop, it may be that the processor time is going to be sucked up
	   no matter what.  You can add a sleep() to the end of your render
	   loop if you would like to yeild to other processes and you don't
	   care too much about the speed of your rendering loop.  If you have
	   Hardware that supports OpenGL (like a 3Dpro card, or GLint card) 
	   then this should be less of a problem, since it won't be rendering
	   in software. (9/11/96)

	o  If a window is fully obscured by another window, the visibility
	   callback is NOT called.  As far as I can tell, this is a limitation
	   of the Win32 api, but a workaround is being searched for. (9/8/96)
	x  Limitation of the Win32 API

	o  Fixed the entry functions.  They only work if the keyboard focus
	   changes.  Therefore, in most Win32 systems, the mouse must be
	   pressed outside of the window to get a GLUT_LEFT message and
	   then pressed inside the window for a GLUT_ENTERED message.

	o  Alt modifier key doesn't work with keyboard callback. (9/8/96)
	x  Probably okay, because the glut spec says that these keys can
	   be intercepted by the system (which the alt key is...) (9/11/96)

	(11/17/96)
	o  glutRemoveMenuItem() not working properly.
	x  Thanks to Gary (grc@maple.civeng.rutgers.edu) for the fix to
	   this one.

	o  Timer functions are messed up.
	x  Thanks to Joseph Galbraith for the fix to this one.

	(12/9/96)
	o  One (minor) difference came up between the X version of glut
	   and the nt one which you should know about. It is not a new
	   problem, and it concerns co-ords returned to the pointer
	   callbacks. (glutMotionFunc, glutMouseFunc)
	   Under X, you get co-ords in the range 0 +/- 2^15, under NT
	   you get 0..2^16. This is only really a problem when moving
	   above or to the left of the window.
	   eg dragging one pixel ABOVE the window will give :-
	   under x11 :      y = -1
	   under nt  :      y = 2^16 -1
	x  Put in fix provided by Shane Clauson.

	(12/17/96)
	o  Idle functions not working properly for multiple windows.
	x  Fixed this by posting an idle message to every window in the 
	   window list when idle.

	(12/18/96)
	o  glutSetCursor() was misbehaving (lthomas@cco.caltech.edu).
	x  Win32 requires that the hCursor member of the window class
           be set to NULL when the class is registered or whenever the
           mouse is moved, the original cursor is replaced (go
           figure!).  Now sets the cursor whenever a WM_MOUSEMOVE message
	   is received, because the WM_SETCURSOR event resets the cursor
	   even when in the decoration area.

	o  Geometry is not being handled quite right.  The numbers don't
	   take into account the window decorations.  That is, if I say
	   make a window 100x100, then the WHOLE window (not just the
	   client area) is 100x100.  Therefore, the client (opengl) area
	   is smaller than 100x100. (9/8/96)
	x  Fixed.  Added code to subtract the decoration size on glutGet()
	   and add the decoration size on glutReshapeWindow().

	o  Multiple glutPostRedisplay() calls are NOT being combined.
	   To get round the "coalesce" problem on glutPostRedisplay,
	   the easiest solution is to roll-your-own coalesce by
	   keeping a global "dirty" flag in the app (eg replace all
	   calls to glutPostRedisplay with image_dirty=TRUE;), and to
	   handle image_dirty with a single glutPostRedisplay in the
	   idle callback when required.  (erk - but increases
	   performance for my particular app (a rendering engine on
	   the end of a pipleine with a stream of graphics updates) by
	   a couple of orders of magnitude ! ) (9/8/96)
	x  Added code to coalesce redisplays.  Every idle cycle, a
	   check is made to see which windows need redisplay, if they
	   need it, a redisplay is posted.  The glutPostRedisplay()
	   call is just a stub that sets a flag.


THANKS:

	Special thanks to the following people:

	Shane Clauson (sc@datamine.co.uk)  [ also provided the GLUT bug icon ]
	Kenny Hoff (hoff@cs.unc.edu)
	Richard Readings (readings@reo.dec.com)
	Paul McQuesten (paulmcq@cs.utexas.edu)
	Philip Winston (winston@cs.unc.edu)	
	JaeWoo Ahn (jaewoo@violet.seri.re.kr)
	Joseph Galbraith (galb@unm.edu)
	Paula Higgins (paulah@unm.edu)
	Sam Fortin (sfortin@searchtech.com)
	Chris Vale (chris.vale@autodesk.com)

	and of course, the original author of GLUT:
	Mark Kilgard (mjk@sgi.com)

	and many others...


COPYRIGHT:

The OpenGL Utility Toolkit distribution for Win32 (Windows NT &
Windows 95) contains source code modified from the original source
code for GLUT version 3.3 which was developed by Mark J. Kilgard.  The
original source code for GLUT is Copyright 1997 by Mark J. Kilgard.
GLUT for Win32 is Copyright 1997 by Nate Robins and is not in the
public domain, but it is freely distributable without licensing fees.
It is provided without guarantee or warrantee expressed or implied.
It was ported with the permission of Mark J. Kilgard by Nate Robins.

THIS SOURCE CODE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OR MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.

OpenGL (R) is a registered trademark of Silicon Graphics, Inc.
