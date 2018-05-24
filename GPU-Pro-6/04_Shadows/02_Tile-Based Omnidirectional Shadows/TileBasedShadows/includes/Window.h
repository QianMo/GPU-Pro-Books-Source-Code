#ifndef WINDOW_H
#define WINDOW_H

// Window
//  
// Simple window class. 
class Window
{
public:
  Window():
    hWnd(NULL),
    hInstance(NULL)
  {
  }

  ~Window()
  {
    Destroy();
  }
  
  // creates window 
  bool Create();

  // destroys window
  void Destroy();

  // handles windows messages
  bool HandleMessages() const;

  HWND GetHWnd() const
  {
    return hWnd;
  }

private:
  HWND hWnd;
  HINSTANCE hInstance;

};

#endif
