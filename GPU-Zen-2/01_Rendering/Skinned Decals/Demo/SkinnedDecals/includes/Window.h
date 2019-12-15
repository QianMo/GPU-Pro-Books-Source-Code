#ifndef WINDOW_H
#define WINDOW_H

// Window
//  
// Simple window class. 
class Window
{
public:
  Window():
    hWnd(nullptr),
    hInstance(nullptr)
  {
  }

  ~Window()
  {
    Destroy();
  }
  
  bool Create();

  void Destroy();

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
