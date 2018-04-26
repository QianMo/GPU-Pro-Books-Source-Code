/***************************************************************************/
/* glKeyAndMouseCallbacks.cpp                                              */
/* -----------------------                                                 */
/*                                                                         */
/* Defines the basic OpenGL/GLUT keyboard and mouse callbacks.  However,   */
/*    since this program allows the file to directly map UI commands to    */
/*    constants, these really shouldn't need frequent changes.  Instead,   */
/*    the calls to ApplyUICommand() defined in glInterface should take     */
/*    care of the heavy lifting and state changes.                         */
/*                                                                         */
/* Chris Wyman (02/01/2008)                                                */
/***************************************************************************/

#include "stencilMultiResSplatting.h"
#include "Scene/Scene.h"
#include "Interface/SceneFileDefinedInteraction.h"
#include "RenderingTechniques/renderingData.h"
#include "DataTypes/Array1D.h"

#define VAR_TYPE_FLOAT   0
#define VAR_TYPE_BOOL    1
#define VAR_TYPE_INT     2

typedef struct _incDecStruct {
	int keyStroke;
	int type;
	int iLimit, iDelta;
	float fLimit, fDelta;
	void *dataPtr;
} incDecStruct;
Array1D< incDecStruct * > incrDecrKeys;


extern Scene *scene;
extern SceneDefinedUI *ui;
extern RenderingData *data;
int trackballInUse = KEY_UNKNOWN, modifiers=0;
int lastMouseX=INT_MAX, lastMouseY=INT_MAX, lastButton=-1;



void ApplyUICommand( unsigned int curCommand, int xVal, int yVal );
bool CheckBoundIncrDecr( unsigned int key );

// These functions search for the correct trackball from above, based
//    on the UI command 'curCommand' and update if appropriately.
//    They are defined in Interface/glTrackballInterface.cpp
void InitializeTrackball( int curCommand, int xVal, int yVal );
void UpdateTrackball( int curCommand, int xVal, int yVal );
void ReleaseTrackball( int curCommand );


void MouseMotionCallback ( int x, int y )
{
	if (lastMouseX == INT_MAX && lastMouseY == INT_MAX) return;

	if (trackballInUse != KEY_UNKNOWN)
		UpdateTrackball( trackballInUse, x, y );
	else // There wasn't a trackball, but maybe there was some other UI functions
	{
		int mouseX = ui->ConvertGLUTMouseX( lastButton, x-lastMouseX, modifiers );
		int mouseY = ui->ConvertGLUTMouseY( lastButton, y-lastMouseY, modifiers );

		ApplyUICommand( mouseX, abs(x-lastMouseX), abs(y-lastMouseY) );
		ApplyUICommand( mouseY, abs(x-lastMouseX), abs(y-lastMouseY) );
	}

	lastMouseX = x;
	lastMouseY = y;
}


void MouseButtonCallback ( int b, int st, int x, int y )
{
	if ( st == GLUT_DOWN )
	{
		lastButton = b;
		lastMouseX = x;
		lastMouseY = y;
		modifiers = glutGetModifiers();
		trackballInUse = ui->ConvertGLUTTrackball( lastButton, modifiers );
		if (trackballInUse) InitializeTrackball( trackballInUse, x, y );
	}
	else
	{
		if (trackballInUse) ReleaseTrackball( trackballInUse );
		lastMouseX = lastMouseY = INT_MAX;
		lastButton = -1;
		modifiers = 0;
		trackballInUse = KEY_UNKNOWN;
	}
}


void KeyboardCallback ( unsigned char key, int x, int y )
{
	UIVariable *var=0;
	unsigned int curKey = ui->ConvertGLUTKey( key );

	// Update any variables bound to this key.  If there were bindings, return.
	if (ui->UpdateBoundVariables( curKey ))
	{
		data->ui->updateHelpScreen = true;
		return;
	}

	// Check if we have a variable bound to this key via the old method
	else if (!CheckBoundIncrDecr( curKey ))
		ApplyUICommand( curKey, x, y );
}


void SpecialKeyboardCallback ( int key, int x, int y )
{
	UIVariable *var=0;
	unsigned int curKey = ui->ConvertGLUTSpecialKey( key );

	// Update any variables bound to this key.  If there were bindings, return.
	if (ui->UpdateBoundVariables( curKey ))
		return;	

	// Check if we have a variable bound to this key via the old method
	else if (!CheckBoundIncrDecr( curKey ))
		ApplyUICommand( curKey, x, y );
}


bool CheckBoundIncrDecr( unsigned int key )
{
	for (unsigned int i=0; i < incrDecrKeys.Size(); i++)
	{
		// Check if there's a match.
		if (key != incrDecrKeys[i]->keyStroke) continue;

		printf("hi!\n");

		// Yep! There's a match
		if (incrDecrKeys[i]->type == VAR_TYPE_BOOL)
		{
			*((bool *)incrDecrKeys[i]->dataPtr) = !*((bool *)incrDecrKeys[i]->dataPtr);
			data->ui->updateHelpScreen = true;
		}
		else if (incrDecrKeys[i]->type == VAR_TYPE_INT)
		{
			*((int *)incrDecrKeys[i]->dataPtr) += incrDecrKeys[i]->iDelta;
			if (incrDecrKeys[i]->iDelta > 0 && *((int *)incrDecrKeys[i]->dataPtr) >= incrDecrKeys[i]->iLimit)
				*((int *)incrDecrKeys[i]->dataPtr) = incrDecrKeys[i]->iLimit;
			if (incrDecrKeys[i]->iDelta < 0 && *((int *)incrDecrKeys[i]->dataPtr) <= incrDecrKeys[i]->iLimit)
				*((int *)incrDecrKeys[i]->dataPtr) = incrDecrKeys[i]->iLimit;
			data->ui->updateHelpScreen = true;
		}
		else if (incrDecrKeys[i]->type == VAR_TYPE_FLOAT)
		{
			*((float *)incrDecrKeys[i]->dataPtr) += incrDecrKeys[i]->fDelta;
			if (incrDecrKeys[i]->fDelta > 0 && *((float *)incrDecrKeys[i]->dataPtr) >= incrDecrKeys[i]->fLimit)
				*((float *)incrDecrKeys[i]->dataPtr) = incrDecrKeys[i]->fLimit;
			if (incrDecrKeys[i]->fDelta < 0 && *((float *)incrDecrKeys[i]->dataPtr) <= incrDecrKeys[i]->fLimit)
				*((float *)incrDecrKeys[i]->dataPtr) = incrDecrKeys[i]->fLimit;
			data->ui->updateHelpScreen = true;
		}
		return true;
	}

	return false;
}

/*
void BindIncrementDecrementKey( char *varName, int key, int by, int limit )
{
	incDecStruct *ids = (incDecStruct *) malloc( sizeof(incDecStruct) );
	ids->keyStroke = key;
	ids->dataPtr = scene->GetSceneIntVar( varName );
	if (ids->dataPtr)
	{
		ids->iDelta = by;
		ids->iLimit = limit;
		ids->type = VAR_TYPE_INT;
		incrDecrKeys.Add( ids );
		return;
	}
	ids->dataPtr = scene->GetSceneFloatVar( varName );
	if (ids->dataPtr)
	{
		ids->fDelta = by;
		ids->fLimit = limit;
		ids->type = VAR_TYPE_FLOAT;
		incrDecrKeys.Add( ids );
		return;
	}
	ids->dataPtr = scene->GetSceneBoolVar( varName );
	if (ids->dataPtr)
	{
		ids->type = VAR_TYPE_BOOL;
		incrDecrKeys.Add( ids );
		return;
	}
}

void BindIncrementDecrementKey( char *varName, int key, float by, float limit )
{
	incDecStruct *ids = (incDecStruct *) malloc( sizeof(incDecStruct) );
	ids->keyStroke = key;
	ids->dataPtr = scene->GetSceneFloatVar( varName );
	if (ids->dataPtr)
	{
		ids->fDelta = by;
		ids->fLimit = limit;
		ids->type = VAR_TYPE_FLOAT;
		incrDecrKeys.Add( ids );
		return;
	}
	ids->dataPtr = scene->GetSceneIntVar( varName );
	if (ids->dataPtr)
	{
		ids->iDelta = by;
		ids->iLimit = limit;
		ids->type = VAR_TYPE_INT;
		incrDecrKeys.Add( ids );
		return;
	}
	ids->dataPtr = scene->GetSceneBoolVar( varName );
	if (ids->dataPtr)
	{
		ids->type = VAR_TYPE_BOOL;
		incrDecrKeys.Add( ids );
		return;
	}
}
*/


