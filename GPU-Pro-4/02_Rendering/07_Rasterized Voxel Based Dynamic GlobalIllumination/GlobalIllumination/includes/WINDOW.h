#ifndef WINDOW_H
#define WINDOW_H

// WINDOW
//  Simple window class. 
class WINDOW
{
public:
	WINDOW()
  {
		hWnd = NULL;
		hInstance = NULL;
	}

	~WINDOW()
	{
		Destroy();
	}
	
	// creates window 
	bool Create();

	// destroys window
	void Destroy();

	// handles windows messages
	bool HandleMessages();

	HWND GetHWnd() const
	{
		return hWnd;
	}

private:
	HWND hWnd;
	HINSTANCE hInstance;

};

#endif
