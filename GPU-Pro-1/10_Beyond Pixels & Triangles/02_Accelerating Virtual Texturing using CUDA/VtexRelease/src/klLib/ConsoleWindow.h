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

/**
	There is only one console for the whole app.
*/
namespace ConsoleWindow {

	/**
		If this character is entered into the console textbox it will close the console.
		This allows you to toggle the console with a single key  ("quake style") if you bind the
		ConsoleWindow::Show function to the same key on your main window.
	    
		Note that this character cannot be typed into the console so ideally you should use
		a system character.
	*/
	static const int CLOSE_CHAR = '`';

	/**
		Implement this class to listen to the user input.
	*/
    class Listener {
    public:
		/**
			Execute the given command text
		*/
        virtual void command(const char *text) {}

		/**
			Called when the console is closed
			Shoud return true if the app wants to quit, otherwise the console window is just hidden.
		*/
        virtual bool canQuit(void) { return false; }
    };

	/**
		Create the console window, sends any typed in commands to the listener
	*/
    bool Create(Listener *listener, const char *titleText = "Console Window" );

	/**
		Free the console window
	*/
	bool Destroy(void);

	/**
		Updates the message loop, this only needs to be called
		if you app does't already pump the message loop.
		Returns false if the application should exit. (WM_QUIT)
	*/
	bool Update(void);

	/**
		Appends text to the console window
	*/
    bool AppendText(const char *text);

	/**
		Show the console flashing in red to the user 
		This never returns and does not let other windows messages trough
		so if you have do do any cleanup do it beforehand...
	*/
	void FatalError(void);

	/**
		Show/Hide the console window
	*/
    bool Show(void);

	/**
		Show/Hide the console window
	*/
    bool Hide(void);

	/**
		Show/Hide the console window
	*/
    bool Toggle(void);

};