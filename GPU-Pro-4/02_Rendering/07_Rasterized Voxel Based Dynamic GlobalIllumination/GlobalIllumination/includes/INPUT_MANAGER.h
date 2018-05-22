#ifndef INPUT_MANAGER_H
#define INPUT_MANAGER_H

#define NUM_KEY_INFOS 256

#define KEY_PRESSED 1       // flag, indicating that key was pressed
#define KEY_QUERIED 2       // flag, indicating that key was queried
#define KEY_MULTI_PRESSED 4 // flag, indicating that key was pressed more than once

// INPUT_MANAGER
//  Simple manager for keyboard-/ mouse-input (based on windows input).
class INPUT_MANAGER
{
public:
  INPUT_MANAGER()
	{
    memset(keyInfos,0,NUM_KEY_INFOS*sizeof(int));
	}

	// gets windows input messages
	bool GetInputMessages(UINT uMsg,WPARAM wParam);

	// updates keyInfos per frame
	void Update();

	// returns true, as long as key pressed 
	bool GetTriggerState(unsigned char keyCode);

	// returns true, exactly once, when key is pressed
	bool GetSingleTriggerState(unsigned char keyCode);

	// gets mouse-position in screen-space
	POINT GetMousePos() const;

	// centers mouse-position in screen-space
	void CenterMousePos() const;

private:
	// sets key-state 
	void SetTriggerState(unsigned char keyCode,bool pressed);

	// array of keyInfos (each representing a bitmask, holding flags) 
	int keyInfos[NUM_KEY_INFOS];

};

#endif