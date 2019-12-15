#include <stdafx.h>
#include <Demo.h>
#include <Window.h>

static LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
  Demo::guiManager->MessageCallback(uMsg, wParam, lParam);

  switch(uMsg)
  {
  // prevent activation of screen saver/ turning monitor off
  case WM_SYSCOMMAND:
    {
      switch(wParam)
      {
      case SC_SCREENSAVE: 
      case SC_MONITORPOWER: 
        return 0;
      }
      break;
    }

  // post quit message, when user tries to close window
  case WM_CLOSE:
    {
      PostQuitMessage(0);
      return 0;
    }

  case WM_KILLFOCUS:
		{
      if(Demo::inputManager)
        Demo::inputManager->ResetKeyInfos();
			return 0;
		}

  case WM_INPUT: 
    {
      if(Demo::inputManager->SetInputMessages(lParam))
        return 0;
    }
  }

  return DefWindowProc(hWnd, uMsg, wParam, lParam);
}

bool Window::Create()
{
  // get handle to the module instance 
  hInstance = (HINSTANCE)GetModuleHandle(nullptr);

  // register window class
  WNDCLASSEX wc = {0};
  wc.cbSize = sizeof(WNDCLASSEX); 
  wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
  wc.lpfnWndProc = (WNDPROC)WndProc;
  wc.hInstance =	hInstance;
  wc.lpszClassName = "Demo"; 
  wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
  if(!RegisterClassEx(&wc))
  {
    LOG_ERROR("Failed to register window class!");
    return false;
  }

  // create window
  DWORD exStyle = WS_EX_APPWINDOW;
  DWORD style = WS_CAPTION | WS_VISIBLE;
  RECT windowRect = {0, 0, SCREEN_WIDTH, SCREEN_HEIGHT}; 
  AdjustWindowRectEx(&windowRect, style, false, exStyle);
  if(!(hWnd = CreateWindowEx(exStyle, "Demo", "Skinned Decals", style,
                             10, 10, windowRect.right-windowRect.left, windowRect.bottom-windowRect.top,
                             nullptr, nullptr, hInstance, nullptr)))
  {
    LOG_ERROR("Failed to create window!");
    Destroy();
    return false;
  }

  ShowWindow(hWnd, SW_SHOW);
  UpdateWindow(hWnd);
  
  return true;
}

void Window::Destroy()
{
  ShowCursor(true);
    
  // destroy window
  if(hWnd)
  {
    if(!DestroyWindow(hWnd))
    {
      LOG_ERROR("Could not destroy window!");
    }
    hWnd = nullptr;
  }
  
  // unregister window class
  UnregisterClass("Demo", hInstance);
  hInstance = nullptr;
}

bool Window::HandleMessages() const
{
  MSG msg;	
  while(PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) 
  {
    if(msg.message == WM_QUIT)
      return false;
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }

  return true;
}

