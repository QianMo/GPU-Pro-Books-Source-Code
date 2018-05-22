
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

#ifndef _PLATFORM_H_
#define _PLATFORM_H_

#if _MSC_VER >= 1400
// To make MSVC 2005 happy
#pragma warning (disable: 4996)
#  define assume(x) __assume(x)
#  define no_alias __declspec(noalias)
#else
#  define assume(x)
#  define no_alias
#endif

// Include some standard files
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define _WIN32_WINNT 0x0500
#  ifndef WINVER
#     define WINVER 0x0500
#  endif
#  include <windows.h>
#else
#  include <sys/time.h>
#  if defined(LINUX)
#    include <X11/keysym.h>
#    include <stdio.h>
#  elif defined(__APPLE__)
#  endif
#endif

//#include <limits.h>
#include <float.h>


// Define some sized types
typedef unsigned char uint8;
typedef   signed char  int8;

typedef unsigned short uint16;
typedef   signed short  int16;

typedef unsigned int uint32;
typedef   signed int  int32;

#include <stddef.h>
typedef ptrdiff_t intptr;

#ifdef _WIN32
	typedef   signed __int64  int64;
	typedef unsigned __int64 uint64;
	typedef LARGE_INTEGER timestamp;
#else
	typedef   signed long long  int64;
	typedef unsigned long long uint64;
	typedef timeval timestamp;
#endif

typedef unsigned char ubyte;
typedef unsigned short ushort;
typedef unsigned int uint;


#ifdef _DEBUG
#  define DEBUG
#endif


// Define some useful macros
#define MCHAR2(a, b) (a | (b << 8))
#define MCHAR4(a, b, c, d) (a | (b << 8) | (c << 16) | (d << 24))

#define _STR_(x) #x
#define DEFINE_STR(x) "#define " #x " " _STR_(x) "\n"

template <typename T, size_t N> char (&ArraySizeHelper(T (&arr)[N]))[N];
#define elementsOf(arr) (sizeof(ArraySizeHelper(arr)))
#define offsetOf(strct, member) uint(((strct *) NULL)->member)

#ifdef _WIN32
#define forceinline __forceinline
#define alignment(x) __declspec(align(x))
#else
#define forceinline inline
#define alignment(x) __attribute__ ((aligned(x)))
#endif

// Utility functions
void ErrorMsg(const char *string);
void WarningMsg(const char *string);
void InfoMsg(const char *string);

// Note: returns true for 0
inline bool isPowerOf2(const int x){
	return (x & (x - 1)) == 0;
}



void initTime();
timestamp getCurrentTime();
float getTimeDifference(const timestamp from, const timestamp to);

#ifdef _WIN32

#ifdef _MSC_VER
// Ensure proper handling of for-scope
#  if (_MSC_VER <= 1200)
#    define for if(0); else for
#  else
#    pragma conform(forScope, on)
#    pragma warning(disable: 4258)
#  endif
#endif

#else

#define stricmp(a, b) strcasecmp(a, b)

#endif // LINUX


#ifdef DEBUG
void outputDebugString(const char *str);
extern void failedAssert(char *file, int line, char *statement);
#define ASSERT(b) if (!(b)) failedAssert(__FILE__, __LINE__, #b)
#else
#define outputDebugString(str)
#define ASSERT(b) assume(b)
#endif // DEBUG



// Platform independent key codes
#define KEY_0 int('0')
#define KEY_1 int('1')
#define KEY_2 int('2')
#define KEY_3 int('3')
#define KEY_4 int('4')
#define KEY_5 int('5')
#define KEY_6 int('6')
#define KEY_7 int('7')
#define KEY_8 int('8')
#define KEY_9 int('9')

#if defined(_WIN32)

#define KEY_LEFT      VK_LEFT
#define KEY_RIGHT     VK_RIGHT
#define KEY_UP        VK_UP
#define KEY_DOWN      VK_DOWN
#define KEY_CTRL      VK_CONTROL
#define KEY_SHIFT     VK_SHIFT
#define KEY_ENTER     VK_RETURN
#define KEY_SPACE     VK_SPACE
#define KEY_TAB       VK_TAB
#define KEY_ESCAPE    VK_ESCAPE
#define KEY_BACKSPACE VK_BACK
#define KEY_HOME      VK_HOME
#define KEY_END       VK_END
#define KEY_INSERT    VK_INSERT
#define KEY_DELETE    VK_DELETE

#define KEY_F1  VK_F1
#define KEY_F2  VK_F2
#define KEY_F3  VK_F3
#define KEY_F4  VK_F4
#define KEY_F5  VK_F5
#define KEY_F6  VK_F6
#define KEY_F7  VK_F7
#define KEY_F8  VK_F8
#define KEY_F9  VK_F9
#define KEY_F10 VK_F10
#define KEY_F11 VK_F11
#define KEY_F12 VK_F12

#define KEY_NUMPAD0 VK_NUMPAD0
#define KEY_NUMPAD1 VK_NUMPAD1
#define KEY_NUMPAD2 VK_NUMPAD2
#define KEY_NUMPAD3 VK_NUMPAD3
#define KEY_NUMPAD4 VK_NUMPAD4
#define KEY_NUMPAD5 VK_NUMPAD5
#define KEY_NUMPAD6 VK_NUMPAD6
#define KEY_NUMPAD7 VK_NUMPAD7
#define KEY_NUMPAD8 VK_NUMPAD8
#define KEY_NUMPAD9 VK_NUMPAD9

#define KEY_ADD        VK_ADD
#define KEY_SUBTRACT   VK_SUBTRACT
#define KEY_MULTIPLY   VK_MULTIPLY
#define KEY_DIVIDE     VK_DIVIDE
#define KEY_SEPARATOR  VK_SEPARATOR
#define KEY_DECIMAL    VK_DECIMAL
#define KEY_PAUSE      VK_PAUSE

#define KEY_A int('A')
#define KEY_B int('B')
#define KEY_C int('C')
#define KEY_D int('D')
#define KEY_E int('E')
#define KEY_F int('F')
#define KEY_G int('G')
#define KEY_H int('H')
#define KEY_I int('I')
#define KEY_J int('J')
#define KEY_K int('K')
#define KEY_L int('L')
#define KEY_M int('M')
#define KEY_N int('N')
#define KEY_O int('O')
#define KEY_P int('P')
#define KEY_Q int('Q')
#define KEY_R int('R')
#define KEY_S int('S')
#define KEY_T int('T')
#define KEY_U int('U')
#define KEY_V int('V')
#define KEY_W int('W')
#define KEY_X int('X')
#define KEY_Y int('Y')
#define KEY_Z int('Z')

#elif defined(LINUX)

#define KEY_LEFT      XK_Left
#define KEY_RIGHT     XK_Right
#define KEY_UP        XK_Up
#define KEY_DOWN      XK_Down
#define KEY_CTRL      XK_Control_R
#define KEY_SHIFT     XK_Shift_R
#define KEY_ENTER     XK_Return
#define KEY_SPACE     XK_space
#define KEY_TAB       XK_Tab
#define KEY_ESCAPE    XK_Escape
#define KEY_BACKSPACE XK_BackSpace
#define KEY_HOME      XK_Home
#define KEY_END       XK_End
#define KEY_INSERT    XK_Insert
#define KEY_DELETE    XK_Delete

#define KEY_F1  XK_F1
#define KEY_F2  XK_F2
#define KEY_F3  XK_F3
#define KEY_F4  XK_F4
#define KEY_F5  XK_F5
#define KEY_F6  XK_F6
#define KEY_F7  XK_F7
#define KEY_F8  XK_F8
#define KEY_F9  XK_F9
#define KEY_F10 XK_F10
#define KEY_F11 XK_F11
#define KEY_F12 XK_F12

#define KEY_NUMPAD0 XK_KP_0
#define KEY_NUMPAD1 XK_KP_1
#define KEY_NUMPAD2 XK_KP_2
#define KEY_NUMPAD3 XK_KP_3
#define KEY_NUMPAD4 XK_KP_4
#define KEY_NUMPAD5 XK_KP_5
#define KEY_NUMPAD6 XK_KP_6
#define KEY_NUMPAD7 XK_KP_7
#define KEY_NUMPAD8 XK_KP_8
#define KEY_NUMPAD9 XK_KP_9

#define KEY_ADD        XK_KP_Add
#define KEY_SUBTRACT   XK_KP_Subtract
#define KEY_MULTIPLY   XK_KP_Multiply
#define KEY_DIVIDE     XK_KP_Divide
#define KEY_SEPARATOR  XK_KP_Separator
#define KEY_DECIMAL    XK_KP_Decimal
#define KEY_PAUSE      XK_Pause

#define KEY_A int('a')
#define KEY_B int('b')
#define KEY_C int('c')
#define KEY_D int('d')
#define KEY_E int('e')
#define KEY_F int('f')
#define KEY_G int('g')
#define KEY_H int('h')
#define KEY_I int('i')
#define KEY_J int('j')
#define KEY_K int('k')
#define KEY_L int('l')
#define KEY_M int('m')
#define KEY_N int('n')
#define KEY_O int('o')
#define KEY_P int('p')
#define KEY_Q int('q')
#define KEY_R int('r')
#define KEY_S int('s')
#define KEY_T int('t')
#define KEY_U int('u')
#define KEY_V int('v')
#define KEY_W int('w')
#define KEY_X int('x')
#define KEY_Y int('y')
#define KEY_Z int('z')

#elif defined(__APPLE__)

#define KEY_LEFT      0x7B
#define KEY_RIGHT     0x7C
#define KEY_UP        0x7E
#define KEY_DOWN      0x7D
#define KEY_CTRL      0x3B
#define KEY_SHIFT     0x38
#define KEY_ENTER     0x24
#define KEY_SPACE     0x31
#define KEY_TAB       0x30
#define KEY_ESCAPE    0x35
#define KEY_BACKSPACE 0x33
#define KEY_HOME      0x73
#define KEY_END       0x77
#define KEY_INSERT    0x72
#define KEY_DELETE    0x33

#define KEY_F1  0x7A
#define KEY_F2  0x78
#define KEY_F3  0x63
#define KEY_F4  0x76
#define KEY_F5  0x60
#define KEY_F6  0x61
#define KEY_F7  0x62
#define KEY_F8  0x64
#define KEY_F9  0x65
#define KEY_F10 0x6D
#define KEY_F11 0x67
#define KEY_F12 0x6F

#define KEY_NUMPAD0 0x52
#define KEY_NUMPAD1 0x53
#define KEY_NUMPAD2 0x54
#define KEY_NUMPAD3 0x55
#define KEY_NUMPAD4 0x56
#define KEY_NUMPAD5 0x57
#define KEY_NUMPAD6 0x58
#define KEY_NUMPAD7 0x59
#define KEY_NUMPAD8 0x5B
#define KEY_NUMPAD9 0x5C

#define KEY_ADD        0x45
#define KEY_SUBTRACT   0x4E
#define KEY_MULTIPLY   0x43
#define KEY_DIVIDE     0x4B
#define KEY_SEPARATOR  0x2B
#define KEY_DECIMAL    0x41
//#define KEY_PAUSE      ????

#define KEY_A 0x00
#define KEY_B 0x0B
#define KEY_C 0x08
#define KEY_D 0x02
#define KEY_E 0x0E
#define KEY_F 0x03
#define KEY_G 0x05
#define KEY_H 0x04
#define KEY_I 0x22
#define KEY_J 0x26
#define KEY_K 0x28
#define KEY_L 0x25
#define KEY_M 0x2E
#define KEY_N 0x2D
#define KEY_O 0x1F
#define KEY_P 0x23
#define KEY_Q 0x0C
#define KEY_R 0x0F
#define KEY_S 0x01
#define KEY_T 0x11
#define KEY_U 0x20
#define KEY_V 0x09
#define KEY_W 0x0D
#define KEY_X 0x07
#define KEY_Y 0x10
#define KEY_Z 0x06

#endif


#endif // _PLATFORM_H_
