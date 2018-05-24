#include <stdafx.h>
#include <Demo.h>
#include <Window.h>

#ifndef UNIX_PORT
// windows procedure
static LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
  // send event message to AntTweakBar
  if(TwEventWin(hWnd, uMsg, wParam, lParam))
    return 0; 

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
  }

  if(Demo::inputManager->GetInputMessages(uMsg, wParam))
    return 0;

  return DefWindowProc(hWnd, uMsg, wParam, lParam);
}
#endif /* UNIX_PORT */

bool Window::Create()
{
#ifndef UNIX_PORT
    
  // get handle to the module instance 
  hInstance = (HINSTANCE)GetModuleHandle(NULL);

  // register window class
  WNDCLASSEX wc = {0};
  wc.cbSize = sizeof(WNDCLASSEX); 
  wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
  wc.lpfnWndProc = (WNDPROC)WndProc;
  wc.hInstance =	hInstance;
  wc.lpszClassName = "Demo"; 
  wc.hCursor = LoadCursor(NULL, IDC_ARROW);
  if(!RegisterClassEx(&wc))
  {
    MessageBox(NULL, "Failed to register window class!", "ERROR", MB_OK | MB_ICONEXCLAMATION);
    return false;
  }

  // create window
  DWORD exStyle = WS_EX_APPWINDOW;
  DWORD style = WS_CAPTION | WS_VISIBLE;
  RECT windowRect = {0, 0, SCREEN_WIDTH, SCREEN_HEIGHT}; 
  AdjustWindowRectEx(&windowRect, style, false, exStyle);
  if(!(hWnd = CreateWindowEx(exStyle, "Demo", "Tile-based Omnidirectional Shadows", style,
                             10, 10, windowRect.right-windowRect.left, windowRect.bottom-windowRect.top,
                             NULL, NULL, hInstance, NULL)))
  {
    MessageBox(NULL, "Failed to create window!", "ERROR", MB_OK | MB_ICONEXCLAMATION);
    Destroy();
    return false;
  }

  ShowWindow(hWnd, SW_SHOW);
  UpdateWindow(hWnd);
    
#else
    
  if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_EVENTS) != 0)
  {
    MessageBox(NULL, "Failed to initialize SDL!", "ERROR", MB_OK | MB_ICONEXCLAMATION);
  }

  hWnd = SDL_CreateWindow("Tile-based Omnidirectional Shadows", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
    SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_INPUT_FOCUS | SDL_WINDOW_MOUSE_FOCUS);
  if(!hWnd)
  {
    MessageBox(NULL, "Failed to create window!", "ERROR", MB_OK | MB_ICONEXCLAMATION);
    Destroy();
    return false;
  }

  atexit(SDL_Quit);
    
#endif /* UNIX_PORT */
  
  return true;
}

void Window::Destroy()
{
  // show cursor
  ShowCursor(true);
    
#ifndef UNIX_PORT

  // destroy window
  if(hWnd)
  {
    if(!DestroyWindow(hWnd))
      MessageBox(NULL, "Could not destroy window!", "ERROR", MB_OK | MB_ICONINFORMATION);
    hWnd = NULL;
  }
  
  // unregister window class
  UnregisterClass("Demo", hInstance);
  hInstance = NULL;
    
#else
    
  if(hWnd)
  {
    SDL_DestroyWindow(hWnd);
    hWnd = nullptr;
  }
    
#endif /* UNIX_PORT */
}

bool Window::HandleMessages() const
{
#ifndef UNIX_PORT

  MSG msg;	
  while(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) 
  {
    if(msg.message == WM_QUIT)
      return false;
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }

#else

  SDL_Event e;
  while(SDL_PollEvent(&e))
  {
    if(TwEventSDL(&e, SDL_MAJOR_VERSION, SDL_MINOR_VERSION) != 0)
    {
      continue;
    }

    if(e.type == SDL_QUIT)
    {
      return false;
    }

    // Forward relevant keys directly to custom input manager.
    switch(e.type)
    {
    case SDL_KEYDOWN:
    case SDL_KEYUP:
      {
        auto key = toupper(e.key.keysym.sym & ~SDLK_SCANCODE_MASK);
        Demo::inputManager->GetInputMessages((e.type == SDL_KEYDOWN) ? WM_KEYDOWN : WM_KEYUP, key);
      }
      break;

    case SDL_MOUSEBUTTONDOWN:
      if(e.button.button == SDL_BUTTON_LEFT || e.button.button == SDL_BUTTON_RIGHT)
      {
        Demo::inputManager->GetInputMessages((e.button.button == SDL_BUTTON_LEFT) ? WM_LBUTTONDOWN : WM_RBUTTONDOWN, 0);
      }
      break;

    case SDL_MOUSEBUTTONUP:
      if(e.button.button == SDL_BUTTON_LEFT || e.button.button == SDL_BUTTON_RIGHT)
      {
        Demo::inputManager->GetInputMessages((e.button.button == SDL_BUTTON_LEFT) ? WM_LBUTTONUP : WM_RBUTTONUP, 0);
      }
      break;
    }
  }

#endif /* UNIX_PORT */

  return true;
}

