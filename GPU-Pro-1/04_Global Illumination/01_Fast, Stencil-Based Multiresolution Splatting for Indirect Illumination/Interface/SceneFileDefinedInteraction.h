
#ifndef SCENE_FILE_DEFINED_INTERACTION_H
#define SCENE_FILE_DEFINED_INTERACTION_H

#include "DataTypes/MathDefs.h"
#include "DataTypes/Point.h"
#include "DataTypes/Vector.h"
#include "DataTypes/Array1D.h"
#include "Utils/TextParsing.h"
#include "UIVars/UIVariable.h"
#include "UIVars/UIInt.h"
#include "UIVars/UIFloat.h"
#include "UIVars/UIBool.h"
#include <stdio.h>
#include "Utils/GLee.h"
#include <GL/glut.h>

#define UI_UNKNOWN							0
#define UI_QUIT								1
#define UI_TRANSLATE_RIGHT					2
#define UI_TRANSLATE_LEFT					3
#define UI_TRANSLATE_UP						4
#define UI_TRANSLATE_DOWN					5
#define UI_TRANSLATE_FORWARD				6
#define UI_TRANSLATE_BACK					7
#define UI_RELOAD_SHADERS					8
#define UI_INCREASE_SAMPLES_PER_PIXEL		9
#define UI_DECREASE_SAMPLES_PER_PIXEL		10
#define UI_DEBUG_RAY						11
#define UI_INCREASE_RAY_DEPTH				12
#define UI_DECREASE_RAY_DEPTH				13
#define UI_INCREASE_FOCAL_LENGTH			14
#define UI_DECREASE_FOCAL_LENGTH			15
#define UI_TOGGLE_PACKET_MODE               16
#define UI_DISPLAY_PIXEL_COSTS              17
#define UI_BRIGHTEN_PIXEL_COSTS				18
#define UI_DARKEN_PIXEL_COSTS				19
#define UI_TOGGLE_BENCHMARKING				20
#define UI_SCREEN_CAPTURE					21
#define UI_TOGGLE_MOVIE_CAPTURE             22
#define UI_ROTATE_UP						23
#define UI_ROTATE_DOWN						24
#define UI_ROTATE_LEFT						25
#define UI_ROTATE_RIGHT						26
#define UI_EYE_TRACKBALL					27
#define UI_LIGHT_TRACKBALL_0				28
#define UI_LIGHT_TRACKBALL_1				29
#define UI_LIGHT_TRACKBALL_2				30
#define UI_LIGHT_TRACKBALL_3				31
#define UI_OBJECT_TRACKBALL_0				32
#define UI_OBJECT_TRACKBALL_1				33
#define UI_OBJECT_TRACKBALL_2				34
#define UI_OBJECT_TRACKBALL_3				35
#define UI_HELPSCREEN						36
#define UI_DEBUG_FRAME                      37
#define UI_OUTPUT_SETTINGS                  38

#define __UI_FUNCTIONS						39


#ifdef __INSIDE_SCENEFILEDEFINEDINTERACTION_CPP_

typedef struct __stringmap {
	int uiID;
	char *string;
} mapping;

mapping uiFuncs[__UI_FUNCTIONS] = {
	{UI_UNKNOWN,						"unknown"},  // Must be first in the list!! 
	{UI_QUIT,							"quit"},     //    Other than UI_UNKNOWN, the 
	{UI_TRANSLATE_RIGHT,				"right"},    //    list can be in any order.
	{UI_TRANSLATE_LEFT,					"left"},
	{UI_TRANSLATE_UP,					"up"},
	{UI_TRANSLATE_DOWN,					"down"},
	{UI_TRANSLATE_FORWARD,				"forward"},
	{UI_TRANSLATE_BACK,					"back"},
	{UI_RELOAD_SHADERS,					"reload-shaders"},
	{UI_INCREASE_SAMPLES_PER_PIXEL,		"more-samples"},
	{UI_DECREASE_SAMPLES_PER_PIXEL,		"less-samples"},
	{UI_DEBUG_RAY,						"debug-ray"},
	{UI_INCREASE_RAY_DEPTH,				"inc-depth"},
	{UI_DECREASE_RAY_DEPTH,				"def-depth"},
	{UI_INCREASE_FOCAL_LENGTH,			"inc-focal-length"},
	{UI_DECREASE_FOCAL_LENGTH,			"dec-focal-length"},
	{UI_TOGGLE_PACKET_MODE,				"toggle-packets"},
	{UI_DISPLAY_PIXEL_COSTS,			"toggle-pixel-cost"},
	{UI_BRIGHTEN_PIXEL_COSTS,			"brighten-pixel-cost"},
	{UI_DARKEN_PIXEL_COSTS,				"darken-pixel-cost"},
	{UI_TOGGLE_BENCHMARKING,			"toggle-benchmark"},
	{UI_SCREEN_CAPTURE,					"screen-capture"},
	{UI_TOGGLE_MOVIE_CAPTURE,			"toggle-movie-capture"},
	{UI_ROTATE_UP,						"rotate-up"},
	{UI_ROTATE_DOWN,					"rotate-down"},
	{UI_ROTATE_LEFT,					"rotate-left"},
	{UI_ROTATE_RIGHT,					"rotate-right"},
	{UI_EYE_TRACKBALL,					"eye-trackball"},
	{UI_LIGHT_TRACKBALL_0,				"light-trackball-0"},
	{UI_LIGHT_TRACKBALL_1,				"light-trackball-1"},
	{UI_LIGHT_TRACKBALL_2,				"light-trackball-2"},
	{UI_LIGHT_TRACKBALL_3,				"light-trackball-3"},
	{UI_OBJECT_TRACKBALL_0,				"obj-trackball-0"},
	{UI_OBJECT_TRACKBALL_1,				"obj-trackball-1"},
	{UI_OBJECT_TRACKBALL_2,				"obj-trackball-2"},
	{UI_OBJECT_TRACKBALL_3,				"obj-trackball-3"},
	{UI_HELPSCREEN,						"help-screen"},
	{UI_DEBUG_FRAME,					"debug-frame"},
	{UI_OUTPUT_SETTINGS,                "output-settings"}
};

#endif

class SceneDefinedUI
{
private:
	int keyMap[__UI_FUNCTIONS];
	Array1D<UIVariable *> varBindings;

	float translateScale[6];
	float rotateScale[4];

	// Takes a string and returns one of the UI_ constants above.
	int MapStringToFunction( char *str ) const;

	// Takes a string and returns one of the KEY_ constants below or
	//    the value str[0].  This allows multi-character strings like
	//    "up-arrow" to map to KEY_UP_ARROW while still allowing "q" to
	//    map to 'q'.  If an unknown multi-character string is input, 
	//    the value of str[0] is returned, though an error message is
	//    emitted if verbose is true.
	int MapStringToKey( char *str, bool verbose=false ) const;

	// Takes an input key (one of the KEY_ constants below and a set of
	//    modifiers (KEY_MODIFIER_*) and returns a constant representing
	//    that key modified by the control keys.  Note multiple modifiers
	//    may be applied by or'ing (|) the KEY_MODIFIER_ constants together.
	int ApplyKeyModifiers( int unmodifiedKey, int modifiers ) const;

	// Input a value returned from MakeStringToKey() to determine if the
	//    string was actually a modifier key (e.g., control, alt, shift)
	bool IsModifier( int key ) const;

	// Given an ASCII string that contains a (possibly modified, i.e., "shift")
	//    representation of a key, convert the string to it's modified ID.
	char *ReadKeyFromBuffer( char *buf, unsigned int *key );
	char *ReadNameAndValue( char *buf, char *name, char *value );
	char *GetNextVariableModifier( char *buf, char *modifier );
	void WarnAndDeleteUnboundVariable( UIVariable *var );
public:
	SceneDefinedUI();           // Set up a default UI mapping
	SceneDefinedUI( FILE *f );  // Use the UI mapping from the specified file
	~SceneDefinedUI() {}

	// Returns the key (unsigned char) associated with a particular UI
	//    function (defined by the UI_ constants above).  The return of
	//    an int allows additional values outside the typical ASCII 
	//    representation.  These are defined by the KEY_ constants.
	// Input of UI_UNKNOWN returns 0, as do UI functionality that has
	//    no associated key.
	// Note:  This class treats mouse and keystrokes identically, but
	//    this may be unclear, so bindings for both Key() and Mouse() are provided.
	inline int Key  ( unsigned int uiFunction ) const { return uiFunction > 0 && uiFunction < __UI_FUNCTIONS ? keyMap[uiFunction] : 0; }
	inline int Mouse( unsigned int uiFunction ) const { return uiFunction > 0 && uiFunction < __UI_FUNCTIONS ? keyMap[uiFunction] : 0; }

	// Returns the bound variable (if it exists)
	bool UpdateBoundVariables( unsigned int key );
	//inline UIVariable *BoundVariable( unsigned int uiFunction ) const { return uiFunction > 0 && uiFunction < __UI_FUNCTIONS ? varBindings[uiFunction] : 0; }

	// Copy all the bound variables into an array.
	void CopyBoundVariables( Array1D<UIVariable *> *varArray );

	inline float GetTranslateScale( int uiFunction ) const { return uiFunction >= UI_TRANSLATE_RIGHT && uiFunction <= UI_TRANSLATE_BACK ? translateScale[uiFunction-UI_TRANSLATE_RIGHT] : 0; }
	inline float GetRotateScale( int uiFunction ) const    { return uiFunction >= UI_ROTATE_UP && uiFunction <= UI_ROTATE_RIGHT ? rotateScale[uiFunction-UI_ROTATE_UP] : 0; }

	// Converts the key passed to the glutKeyFunc() or glutSpecialFunc() to 
	//    value used by this class.  It also takes into accound GLUT modifiers.
	int ConvertGLUTKey( unsigned char key, int modifiers=glutGetModifiers() ) const;
	int ConvertGLUTSpecialKey( int key, int modifiers=glutGetModifiers() ) const;
	int ConvertGLUTModifiers( int modifiers=glutGetModifiers() ) const;
	int ConvertGLUTMouseX( int button, int deltaX, int modifiers=glutGetModifiers() ) const;
	int ConvertGLUTMouseY( int button, int deltaY, int modifiers=glutGetModifiers() ) const;

	// Returns a key *if* there is a valid trackball associated with the current 
	//    button + modifier combination.  Otherwise, it returns KEY_UNKNOWN and the
	//    ConvertGLUTMouseX and ConvertGLUTMouseY functions should be used instead.
	int ConvertGLUTTrackball( int button, int modifiers ) const;
	
	

};



// Key for all unknown keys
#define KEY_UNKNOWN         0

// Control characters (these are ASCII)
#define KEY_ESCAPE			27
#define KEY_TAB				9
#define KEY_RETURN			13
#define KEY_BACKSPACE       8
#define KEY_DELETE			127

// Special (usually non-ASCII keys)
#define KEY_UP_ARROW		1001
#define KEY_DOWN_ARROW		1002
#define KEY_LEFT_ARROW		1003
#define KEY_RIGHT_ARROW		1004
#define KEY_F1				1005
#define KEY_F2				1006
#define KEY_F3				1007
#define KEY_F4				1008
#define KEY_F5				1009
#define KEY_F6				1010
#define KEY_F7				1011
#define KEY_F8				1012
#define KEY_F9				1013
#define KEY_F10				1014
#define KEY_F11				1015
#define KEY_F12				1016
#define KEY_INSERT			1017
#define KEY_HOME			1019
#define KEY_END				1020
#define KEY_PAGE_UP			1021
#define KEY_PAGE_DOWN		1022

#define KEY_NO_MODIFIER			0x00000000
#define KEY_UNMODIFIED_MASK		0x0FFFFFFF
#define KEY_MODIFIER_MASK       0x70000000
#define KEY_MODIFIER_SHIFT		0x10000000
#define KEY_MODIFIER_CONTROL	0x20000000
#define KEY_MODIFIER_ALT		0x40000000

#define MOUSE_BASE               5000
#define MOUSE_NOBUTTON_POS_X	 5000
#define MOUSE_NOBUTTON_NEG_X	 5001
#define MOUSE_NOBUTTON_POS_Y	 5002
#define MOUSE_NOBUTTON_NEG_Y	 5003
#define MOUSE_LBUTTON_POS_X      5004
#define MOUSE_LBUTTON_NEG_X      5005
#define MOUSE_LBUTTON_POS_Y      5006
#define MOUSE_LBUTTON_NEG_Y      5007
#define MOUSE_MBUTTON_POS_X      5008
#define MOUSE_MBUTTON_NEG_X      5009
#define MOUSE_MBUTTON_POS_Y      5010
#define MOUSE_MBUTTON_NEG_Y      5011
#define MOUSE_RBUTTON_POS_X      5012
#define MOUSE_RBUTTON_NEG_X      5013
#define MOUSE_RBUTTON_POS_Y      5014
#define MOUSE_RBUTTON_NEG_Y      5015

#define MOUSE_LBUTTON            5025
#define MOUSE_MBUTTON            5026
#define MOUSE_RBUTTON            5027


#ifdef __INSIDE_SCENEFILEDEFINEDINTERACTION_CPP_

// Defines the strings from the file that map to each of the special keys.
//   Of course, single ASCII characters (e.g. 'a' or 'f' without the quotes)
//   are also valid identifiers.  These need not be added here.  Please note
//   that 'Q' is not a valid identifier.  The correct identifier for that is
//   "shift q" where "shift" is the modifier from the list below.

#define NUM_KEYSTRINGS   49
mapping keyStrings[NUM_KEYSTRINGS] = {
	{KEY_UNKNOWN,					"unknown"},  // Must be first in the list!!
	{KEY_ESCAPE,					"escape"},
	{KEY_TAB,						"tab"},
	{KEY_RETURN,					"return"},
	{KEY_BACKSPACE,					"backspace"},
	{KEY_DELETE,					"delete"},
	{KEY_MODIFIER_SHIFT,			"shift"},
	{KEY_MODIFIER_CONTROL,			"control"},
	{KEY_MODIFIER_ALT,				"alt"},
	{KEY_UP_ARROW,					"up-arrow"},
	{KEY_DOWN_ARROW,				"down-arrow"},
	{KEY_LEFT_ARROW,				"left-arrow"},
	{KEY_RIGHT_ARROW,				"right-arrow"},
	{KEY_F1,						"f1"},
	{KEY_F2,						"f2"},
	{KEY_F3,						"f3"},
	{KEY_F4,						"f4"},
	{KEY_F5,						"f5"},
	{KEY_F6,						"f6"},
	{KEY_F7,						"f7"},
	{KEY_F8,						"f8"},
	{KEY_F9,						"f9"},
	{KEY_F10,						"f10"},
	{KEY_F11,						"f11"},
	{KEY_F12,						"f12"},
	{KEY_INSERT,					"insert"},
	{KEY_HOME,						"home"},
	{KEY_END,						"end"},
	{KEY_PAGE_UP,					"page-up"},
	{KEY_PAGE_DOWN,					"page-down"},
	{MOUSE_NOBUTTON_POS_X,			"mouse-move-pos-x"},
	{MOUSE_NOBUTTON_NEG_X,			"mouse-move-neg-x"},
	{MOUSE_NOBUTTON_POS_Y,			"mouse-move-pos-y"},
	{MOUSE_NOBUTTON_NEG_Y,			"mouse-move-neg-y"},
	{MOUSE_LBUTTON_POS_X,			"mouse-left-pos-x"},
	{MOUSE_LBUTTON_NEG_X,			"mouse-left-neg-x"},
	{MOUSE_LBUTTON_POS_Y,			"mouse-left-pos-y"},
	{MOUSE_LBUTTON_NEG_Y,			"mouse-left-neg-y"},
	{MOUSE_MBUTTON_POS_X,			"mouse-middle-pos-x"},
	{MOUSE_MBUTTON_NEG_X,			"mouse-middle-neg-x"},
	{MOUSE_MBUTTON_POS_Y,			"mouse-middle-pos-y"},
	{MOUSE_MBUTTON_NEG_Y,			"mouse-middle-neg-y"},
	{MOUSE_RBUTTON_POS_X,			"mouse-right-pos-x"},
	{MOUSE_RBUTTON_NEG_X,			"mouse-right-neg-x"},
	{MOUSE_RBUTTON_POS_Y,			"mouse-right-pos-y"},
	{MOUSE_RBUTTON_NEG_Y,			"mouse-right-neg-y"},
	{MOUSE_LBUTTON,					"mouse-left"},
	{MOUSE_MBUTTON,					"mouse-middle"},
	{MOUSE_RBUTTON,					"mouse-right"}
};

#endif


#endif

