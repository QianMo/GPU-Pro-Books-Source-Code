#pragma once

/*
Copyright (c) 2005-2009 Charles-Frederik Hollemeersch

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
*/

#define USES_GLEW

namespace RenderWindow {

    class Listener {
    public:

        enum MouseButtonType {
            MOUSE_LEFT,
            MOUSE_RIGHT,
            MOUSE_MIDDLE
        };

        /**
            This will be called when an error occurs in the renderwindow system.
        */
        virtual void error(const char *text) {}

        /**
            This will be called when opengl & glew has been properly initialized
        */
        virtual bool glInit(void) {return true;}
        virtual bool glFree(void) {return true;}

        /**
            The window has been resized, allso called after the window is created
            to inform about the created window size
        */
        virtual void resize(size_t width, size_t height, float aspectRatio) {}

        /**
            A key has been pressed
        */
        virtual void keyDown(size_t key) {}

        /**
            A key has been released
        */
        virtual void keyUp(size_t key) {}

        /**
            A character has been typed (translated key)
        */
        virtual void onChar(int theChar) {}

        /**
            A mouse button has been pressed
        */
        virtual void mouseDown(MouseButtonType btn) {}

        /**
            A  mouse button has been released
        */
        virtual void mouseUp(MouseButtonType btn) {}

        /**
            The mouse wheel has moved
        */
        virtual void mouseWheel(int delta) {}

        /**
            The mouse has been moved
        */
        virtual void mouseMove(int x, int y) {}

        /**
            Returns true if we allow the render window to close now.
        */
        virtual bool canClose(void) { return true; }
    };

    /**
        Open the window
    */
    bool Open(Listener *listener,
              const char *title = "RenderWindow",
              int width=640, int height=480,
              bool fullscreen=false, int bpp=32);

    /**
        Close the window, call at shutdown
    */
    void Close(void);

    /**
        Processes messages (this will call the listener functions)
        Returns false if the application main loop should quit
    */
    bool Update(void);

    /**
        Swap the front/and backbuffers of opengl
    */
    bool SwapBuffers(void);

}