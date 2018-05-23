//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//--------------------------------------------------------------------------------------
#ifndef __CPUTEVENTHANDLER_H__
#define __CPUTEVENTHANDLER_H__

#include <stdio.h>

// event handling enums
//-----------------------------------------------------------------------------
enum CPUTKey
{
    KEY_NONE,

    // caps keys
    KEY_A,
    KEY_B,
    KEY_C,
    KEY_D,
    KEY_E,
    KEY_F,
    KEY_G,
    KEY_H,
    KEY_I,
    KEY_J,
    KEY_K,
    KEY_L,
    KEY_M,
    KEY_N,
    KEY_O,
    KEY_P,
    KEY_Q,
    KEY_R,
    KEY_S,
    KEY_T,
    KEY_U,
    KEY_V,
    KEY_W,
    KEY_X,
    KEY_Y,
    KEY_Z,

    // number keys
    KEY_1,
    KEY_2,
    KEY_3,
    KEY_4,
    KEY_5,
    KEY_6,
    KEY_7,
    KEY_8,
    KEY_9,
    KEY_0,

    // symbols
    KEY_SPACE,
    KEY_BACKQUOTE,
    KEY_TILDE,
    KEY_EXCLAMATION,
    KEY_AT,
    KEY_HASH,
    KEY_$,
    KEY_PERCENT,
    KEY_CARROT,
    KEY_ANDSIGN,
    KEY_STAR,
    KEY_OPENPAREN,
    KEY_CLOSEPARN,
    KEY__,
    KEY_MINUS,
    KEY_PLUS,

    KEY_OPENBRACKET,
    KEY_CLOSEBRACKET,
    KEY_OPENBRACE,
    KEY_CLOSEBRACE,
    KEY_BACKSLASH,
    KEY_PIPE,
    KEY_SEMICOLON,
    KEY_COLON,
    KEY_SINGLEQUOTE,
    KEY_QUOTE,
    KEY_COMMA,
    KEY_PERIOD,
    KEY_SLASH,
    KEY_LESS,
    KEY_GREATER,
    KEY_QUESTION,

    // function keys
    KEY_F1,
    KEY_F2,
    KEY_F3,
    KEY_F4,
    KEY_F5,
    KEY_F6,
    KEY_F7,
    KEY_F8,
    KEY_F9,
    KEY_F10,
    KEY_F11,
    KEY_F12,

    // special keys
    KEY_HOME,
    KEY_END,
    KEY_INSERT,
    KEY_DELETE,
    KEY_PAGEUP,
    KEY_PAGEDOWN,

    KEY_UP,
    KEY_DOWN,
    KEY_LEFT,
    KEY_RIGHT,

    KEY_BACKSPACE,
    KEY_ENTER,
    KEY_TAB,
    KEY_PAUSE,
    KEY_CAPSLOCK,
    KEY_ESCAPE,

    // control keys
    KEY_LEFT_SHIFT,
    KEY_RIGHT_SHIFT,
    KEY_LEFT_CTRL,
    KEY_RIGHT_CTRL,
    KEY_LEFT_ALT,
    KEY_RIGHT_ALT,
};

// these must be unique because we bitwise && them to get multiple states
enum CPUTMouseState
{
    CPUT_MOUSE_NONE = 0,
    CPUT_MOUSE_LEFT_DOWN = 1,
    CPUT_MOUSE_RIGHT_DOWN = 2,
    CPUT_MOUSE_MIDDLE_DOWN = 4,
    CPUT_MOUSE_CTRL_DOWN = 8,
    CPUT_MOUSE_SHIFT_DOWN = 16,
    CPUT_MOUSE_WHEEL = 32,
};

enum CPUTEventHandledCode
{
    CPUT_EVENT_HANDLED = 0,
    CPUT_EVENT_UNHANDLED = 1,
    CPUT_EVENT_PASSTHROUGH = 2,
};


// Event handler interface - used by numerous classes in the system
class CPUTEventHandler
{
public:
    virtual CPUTEventHandledCode HandleKeyboardEvent(CPUTKey key)=0;
    virtual CPUTEventHandledCode HandleMouseEvent(int x, int y, int wheel, CPUTMouseState state)=0;
};


#endif //#ifndef __CPUTEVENTHANDLER_H__