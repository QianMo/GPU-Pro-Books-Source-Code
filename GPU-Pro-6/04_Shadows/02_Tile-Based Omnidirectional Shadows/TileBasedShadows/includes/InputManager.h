#ifndef INPUT_MANAGER_H
#define INPUT_MANAGER_H

#define NUM_KEY_INFOS 256

// InputManager
//
// Simple manager for keyboard-/ mouse-input (based on windows input).
class InputManager
{
public:
  InputManager();

  // gets windows input messages
  bool GetInputMessages(UINT uMsg, WPARAM wParam);

  // updates keyInfos per frame
  void Update();

  // returns true, as long as key pressed 
  bool GetTriggerState(size_t keyCode);

  // returns true, exactly once, when key is pressed
  bool GetSingleTriggerState(size_t keyCode);

  // gets mouse-position in screen-space
  POINT GetMousePos(bool windowSpace=false) const;

  // sets mouse-position in screen-space
  void SetMousePos(POINT position, bool windowSpace=false) const;

  // toggle mouse-cursor
  void ShowMouseCursor(bool show);

private:
  // sets key-state 
  void SetTriggerState(size_t keyCode, bool pressed);

  // array of keyInfos (each representing a bitmask, holding flags) 
  unsigned int keyInfos[NUM_KEY_INFOS];

  bool cursorVisible;

};

#endif