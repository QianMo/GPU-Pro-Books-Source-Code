#ifndef INPUT_MANAGER_H
#define INPUT_MANAGER_H

#define NUM_KEY_INFOS 256

// InputManager
//
// Simple manager for keyboard-/ mouse-input (based on windows input).
class InputManager
{
public:
  InputManager():
    cursorVisible(true)
  {
    ResetKeyInfos();
  }

  void ResetKeyInfos();

  bool Init();

  // sets windows input messages
  bool SetInputMessages(LPARAM lParam);

  // updates keyInfos per frame
  void Update();

  // returns true, as long as key is pressed 
  bool GetTriggerState(size_t keyCode);

  // returns true, exactly once, when key is pressed
  bool GetSingleTriggerState(size_t keyCode);

  // gets mouse-position in screen-space
  POINT GetMousePos() const;

  // sets mouse-position in screen-space
  void SetMousePos(POINT position);

  // toggle mouse-cursor
  void ShowMouseCursor(bool show);

private:
  // sets key-state 
  void SetTriggerState(size_t keyCode, bool pressed);

  // array of keyInfos (each representing a bitmask, holding flags) 
  UINT keyInfos[NUM_KEY_INFOS];

  POINT mousePos;
  bool cursorVisible;

};

#endif