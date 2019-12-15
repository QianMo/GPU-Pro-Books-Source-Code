#include <stdafx.h>
#include <Demo.h>
#include <InputManager.h>

#define KEY_PRESSED 1       // flag, indicating that key was pressed
#define KEY_QUERIED 2       // flag, indicating that key was queried
#define KEY_MULTI_PRESSED 4 // flag, indicating that key was pressed more than once

void InputManager::ResetKeyInfos()
{
  memset(keyInfos, 0, NUM_KEY_INFOS * sizeof(UINT));  
}

bool InputManager::Init()
{
  // register mouse/ keyboard raw input devices
  RAWINPUTDEVICE Rid[2]; 
  Rid[0].usUsagePage = 0x01; 
  Rid[0].usUsage = 0x02; 
  Rid[0].dwFlags = 0;
  Rid[0].hwndTarget = Demo::window->GetHWnd();
  Rid[1].usUsagePage = 0x01; 
  Rid[1].usUsage = 0x06; 
  Rid[1].dwFlags = 0;	
  Rid[1].hwndTarget = Demo::window->GetHWnd();
  if(RegisterRawInputDevices(Rid, 2, sizeof(Rid[0])) == FALSE) 
  { 
  	LOG_ERROR("Failed to register raw input devices"); 
    return false;
  } 

  // init mouse position
  GetCursorPos(&mousePos);

  return true;
}

bool InputManager::SetInputMessages(LPARAM lParam)
{ 
  BYTE pData[128]; 
  UINT dataSize;
  GetRawInputData((HRAWINPUT)lParam, RID_INPUT, pData, &dataSize, sizeof(RAWINPUTHEADER)); 

  RAWINPUT *rawInput = (RAWINPUT*)pData; 
  if(rawInput->header.dwType == RIM_TYPEMOUSE) 
  { 
    // mouse position changed
    if(rawInput->data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE)
    {
      mousePos.x = rawInput->data.mouse.lLastX; 
      mousePos.y = rawInput->data.mouse.lLastY; 
    }
    else
    {
      mousePos.x += rawInput->data.mouse.lLastX; 
      mousePos.y += rawInput->data.mouse.lLastY; 
    }

    // left mouse-button is pressed down
    if(rawInput->data.mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_DOWN)
      SetTriggerState(VK_LBUTTON, true);

    // left mouse-button is released
    if(rawInput->data.mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_UP)
      SetTriggerState(VK_LBUTTON, false);

    // right mouse-button is pressed down
    if(rawInput->data.mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_DOWN)
      SetTriggerState(VK_RBUTTON, true);

    // right mouse-button is released
    if(rawInput->data.mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_UP)
      SetTriggerState(VK_RBUTTON, false);

    return true;
  }
  else if(rawInput->header.dwType == RIM_TYPEKEYBOARD)
  {
    // keyboard key is pressed down  
    if(rawInput->data.keyboard.Message == WM_KEYDOWN)
       SetTriggerState(rawInput->data.keyboard.VKey, true);
    
    // keyboard key is released
    if(rawInput->data.keyboard.Message == WM_KEYUP)
      SetTriggerState(rawInput->data.keyboard.VKey, false);

    return true;
  }

  return false;
}

void InputManager::Update()
{
  for(UINT i=0; i<NUM_KEY_INFOS; i++)
  {
    if(keyInfos[i] & KEY_QUERIED)
      keyInfos[i] |= KEY_MULTI_PRESSED;
  }
}

void InputManager::SetTriggerState(size_t keyCode, bool pressed)
{
  if(keyCode >= NUM_KEY_INFOS)
    return;

  if(pressed)
  {
    keyInfos[keyCode] |= KEY_PRESSED;
  }
  else
  {
    keyInfos[keyCode] &= ~KEY_PRESSED;
    keyInfos[keyCode] &= ~KEY_QUERIED;
    keyInfos[keyCode] &= ~KEY_MULTI_PRESSED;
  }
}

bool InputManager::GetTriggerState(size_t keyCode)
{
  if(keyCode >= NUM_KEY_INFOS)
    return false;
  return (keyInfos[keyCode] & KEY_PRESSED);
}

bool InputManager::GetSingleTriggerState(size_t keyCode)
{
  if(keyCode >= NUM_KEY_INFOS)
    return false;
  if((keyInfos[keyCode] & KEY_PRESSED) && (!(keyInfos[keyCode] & KEY_MULTI_PRESSED)))
  {
    keyInfos[keyCode] |= KEY_QUERIED;
    return true;
  }
  return false; 
}

POINT InputManager::GetMousePos() const
{ 
  return mousePos;
}

void InputManager::SetMousePos(POINT position)
{
  SetCursorPos(position.x, position.y);
  mousePos.x = position.x;
  mousePos.y = position.y;
}

void InputManager::ShowMouseCursor(bool show) 
{
  if(cursorVisible == show)
    return;

  int counter;
  if(show)
  {
    do
    {
      counter = ShowCursor(true);
    }
    while(counter < 0);
  }
  else
  {
    do
    {
      counter = ShowCursor(false);
    }
    while(counter >= 0);
  }

  cursorVisible = show;
}

