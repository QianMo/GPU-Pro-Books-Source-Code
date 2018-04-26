


#define __INSIDE_SCENEFILEDEFINEDINTERACTION_CPP_
#include "SceneFileDefinedInteraction.h"
#undef  __INSIDE_SCENEFILEDEFINEDINTERACTION_CPP_

// Include for FatalError()...
#include "Utils/ImageIO/imageIO.h"
#include "DataTypes/MathDefs.h"


void BindIncrementDecrementKey( char *varName, int key, int by, int limit );
void BindIncrementDecrementKey( char *varName, int key, float by, float limit );

SceneDefinedUI::SceneDefinedUI() 
{
	memset( keyMap, UI_UNKNOWN, __UI_FUNCTIONS * sizeof( int ) );
	translateScale[0] = translateScale[1] = translateScale[2] = 1;
	translateScale[3] = translateScale[4] = translateScale[5] = 1;
	rotateScale[0] = rotateScale[1] = rotateScale[2] = rotateScale[3] = 1;

	// Setup default key mappings here.
	keyMap[UI_QUIT		] = KEY_ESCAPE; 
}

char *SceneDefinedUI::ReadKeyFromBuffer( char *buf, unsigned int *key )
{
	// Read in the key....
	int _key=-1, modifiers = KEY_NO_MODIFIER;
	bool isModifier, goodData;
	char *ptr = buf, token[256];
	ptr = StripLeadingTokenToBuffer( ptr, token );
	MakeLower( token );
	do
	{
		goodData = ptr[0] > 0;
		_key = MapStringToKey( token, true );
		isModifier = IsModifier( _key );
		if (isModifier) 
		{
			modifiers |= _key;
			ptr = StripLeadingTokenToBuffer( ptr, token );
			MakeLower( token );
		}
	} while (isModifier && goodData);
	_key = ApplyKeyModifiers( _key, modifiers );
	_key = _key > 0 ? _key : KEY_UNKNOWN;
	*key = _key;
	return ptr;
}

char *SceneDefinedUI::GetNextVariableModifier( char *buf, char *modifier )
{
	char token[256];
	char *ptr = StripLeadingTokenToBuffer( buf, token );
	MakeLower( token );
	do
	{
		if (!strcmp(token, "toggle")) break;
		if (!strcmp(token, "by"))     break;
		if (!strcmp(token, "incr"))   break;
		if (!strcmp(token, "decr"))   break;
		if (!strcmp(token, "range"))  break;
		ptr = StripLeadingTokenToBuffer( ptr, token );
		MakeLower( token );
	} while ( ptr[0] );
	strcpy( modifier, token );
	return ptr;
}


char *SceneDefinedUI::ReadNameAndValue( char *buf, char *name, char *value )
{
	char *ptr = buf;
	// We found a declaration for a UIBool.  Find its name
	ptr = StripLeadingTokenToBuffer( ptr, name );	
	MakeLower( name );

	// Get the value
	ptr = StripLeadingTokenToBuffer( ptr, value );	
	MakeLower( value );
	if (value[0] == 'i')  // we actually read "is"
	{                     // so get the token after, which has the value
		ptr = StripLeadingTokenToBuffer( ptr, value );	
		MakeLower( value );
	}

	return ptr;
}

void SceneDefinedUI::WarnAndDeleteUnboundVariable( UIVariable *var )
{
	printf("  (-) Warning! Variable '%s' not bound to any keystroke!\n", var->GetName()); 
	delete var;
}

SceneDefinedUI::SceneDefinedUI( FILE *f ) 
{
	memset( keyMap, UI_UNKNOWN, __UI_FUNCTIONS * sizeof( int ) );
	translateScale[0] = translateScale[1] = translateScale[2] = 1;
	translateScale[3] = translateScale[4] = translateScale[5] = 1;
	rotateScale[0] = rotateScale[1] = rotateScale[2] = rotateScale[3] = 1;

	// Search the scene file.
	char buf[ MAXLINELENGTH ], token[256], *ptr, name[256], val[256];
	while( fgets(buf, MAXLINELENGTH, f) != NULL )  
	{
		// Is this line a comment?
		ptr = StripLeadingWhiteSpace( buf );
		if (ptr[0] == '#' || ptr[0] == '\n' || ptr[0] == '\r' || ptr[0] == 0 ) continue;

		// Nope.  So find out what the command is...
		ptr = StripLeadingTokenToBuffer( ptr, token );
		MakeLower( token );
	
		// Take different measures, depending on the command.
		if (!strcmp(token,"end")) break;
		else if (!strcmp(token,"scale"))
		{
			// First read the function to map to
			ptr = StripLeadingTokenToBuffer( ptr, token );
			int func = MapStringToFunction( token );

			// Now find the scaling
			if ( func >= UI_TRANSLATE_RIGHT && func <= UI_TRANSLATE_BACK )
			{
				ptr = StripLeadingTokenToBuffer( ptr, token );
				translateScale[func-UI_TRANSLATE_RIGHT] = atof( token );
			}
			else if ( func >= UI_ROTATE_UP && func <= UI_ROTATE_RIGHT )
			{
				ptr = StripLeadingTokenToBuffer( ptr, token );
				rotateScale[func-UI_ROTATE_UP] = atof( token );
			}
		}
		else if (!strcmp(token,"map"))
		{
			// First read the function to map to
			ptr = StripLeadingTokenToBuffer( ptr, token );
			int func = MapStringToFunction( token );

			// Read the key (and modifiers) to map to that function
			//    If the first (or second or third) word is a modifer (e.g., alt
			//    control, shift), add it to the modifier list and keep looking for
			//    the actual key to bind.
			int key=-1, modifiers = KEY_NO_MODIFIER;
			bool isModifier;
			do
			{
				ptr = StripLeadingTokenToBuffer( ptr, token );
				key = MapStringToKey( token, true );
				isModifier = IsModifier( key );
				if (isModifier) modifiers |= key;
			} while (isModifier && ptr[0]);
			key = ApplyKeyModifiers( key, modifiers );

			// Assign the relevent key to the appropriate function.
			//    Note keyMap[0] is never used, so this still works 
			//    should MapStringToFunction() return UI_UNKNOWN.
			keyMap[func] = key > 0 ? key : KEY_UNKNOWN;
		}
		else if (!strcmp(token,"bool"))
		{
			// We found a declaration for a UIBool.  Find its name and value
			bool bound = false;
			ptr = ReadNameAndValue( ptr, name, val );

			// Create our variable
			bool value = val[0] == 't' ? true : false;
			UIBool *bVar = new UIBool( value, name );
			//printf("Allocated boolean variable '%s' with value '%s'...\n", name, val );

			// Grab details about binding
			while ( ptr[0] != 0 && 
				    (ptr = GetNextVariableModifier(ptr, token)) ) 
			{
				if (!strcmp(token,"toggle"))
				{
					// All right.  We found a binding.  So set it up.
					unsigned int key;
					ptr = ReadKeyFromBuffer( ptr, &key );
					bVar->AddKeyResponse( key );
					//printf("  (-) Toggle bound to key value %d\n", key );
					varBindings.Add( bVar );
					bound = true;
				}
			}
			if (!bound) WarnAndDeleteUnboundVariable(bVar);
			continue;
		}
		else if (!strcmp(token,"float"))
		{
			// We found a declaration for a UIBool.  Find its name and value
			bool bound = false;
			float increment = 0.1, fMax = FLT_MAX, fMin = FLT_MIN;
			ptr = ReadNameAndValue( ptr, name, val );

			// Create our variable
			float value = atof( val );
			UIFloat *fVar = new UIFloat( value, name );
			//printf("Allocated float variable '%s' with value '%s'...\n", name, val );

			// Grab details about binding
			while ( ptr[0] != 0 && 
				    (ptr = GetNextVariableModifier(ptr, token)) ) 
			{
				if (!strcmp(token,"by"))
					ptr = StripLeadingNumber( ptr, &increment );
				else if (!strcmp(token,"range"))
				{
					ptr = StripLeadingNumber( ptr, &fMin );
					ptr = StripLeadingNumber( ptr, &fMax );
					fVar->SetValueRange( fMin, fMax );
				}
				else if (!strcmp(token,"incr"))
				{
					// All right.  We found a binding.  So set it up.
					unsigned int key;
					ptr = ReadKeyFromBuffer( ptr, &key );
					fVar->AddKeyResponse( key, increment );
					//printf("  (-) Increment bound to key value %d\n", key );
					bound = true;
				}
				else if (!strcmp(token,"decr"))
				{
					// All right.  We found a binding.  So set it up.
					unsigned int key;
					ptr = ReadKeyFromBuffer( ptr, &key );
					fVar->AddKeyResponse( key, -increment );
					//printf("  (-) Decrement bound to key value %d\n", key );
					bound = true;
				}
			}
			if (!bound) WarnAndDeleteUnboundVariable(fVar);
			varBindings.Add( fVar );
			continue;
		}
		else if (!strcmp(token,"int"))
		{
			// We found a declaration for a UIInt.  Find its name and value
			bool bound = false;
			int increment = 1, iMax = INT_MAX, iMin = INT_MIN;
			ptr = ReadNameAndValue( ptr, name, val );

			// Create our variable
			int value = atoi( val );
			UIInt *iVar = new UIInt( value, name );
			//printf("Allocated int variable '%s' with value '%s'...\n", name, val );

			// Grab details about binding
			while ( ptr[0] != 0 && 
				    (ptr = GetNextVariableModifier(ptr, token)) ) 
			{
				if (!strcmp(token,"by"))
				{
					ptr = StripLeadingTokenToBuffer( ptr, token );
					increment = atoi( token );
				}
				else if (!strcmp(token,"range"))
				{
					ptr = StripLeadingTokenToBuffer( ptr, token );
					iMin = atoi( token );
					ptr = StripLeadingTokenToBuffer( ptr, token );
					iMax = atoi( token );
					iVar->SetValueRange( iMin, iMax );
				}
				else if (!strcmp(token,"incr"))
				{
					// All right.  We found a binding.  So set it up.
					unsigned int key;
					ptr = ReadKeyFromBuffer( ptr, &key );
					iVar->AddKeyResponse( key, increment );
					//printf("  (-) Increment bound to key value %d\n", key );
					bound = true;
				}
				else if (!strcmp(token,"decr"))
				{
					// All right.  We found a binding.  So set it up.
					unsigned int key;
					ptr = ReadKeyFromBuffer( ptr, &key );
					iVar->AddKeyResponse( key, -increment );
					//printf("  (-) Decrement bound to key value %d\n", key );
					bound = true;
				}
			}
			if (!bound) WarnAndDeleteUnboundVariable(iVar);
			varBindings.Add( iVar );
			continue;
		}
		else
			printf("Error: Unknown command '%s' when loading SceneDefinedUI!\n", token);
	}


}

bool SceneDefinedUI::UpdateBoundVariables( unsigned int key )
{
	bool modified = false;
	for (unsigned int i=0; i<varBindings.Size(); i++)
	{
		modified = modified || (varBindings[i]->UpdateFromInput( key ));
	}
	return modified;
}


void SceneDefinedUI::CopyBoundVariables( Array1D<UIVariable *> *varArray )
{
	for (unsigned int i=0; i<varBindings.Size(); i++)
		varArray->Add( varBindings[i] );
}


int SceneDefinedUI::MapStringToFunction( char *str ) const
{
	MakeLower( str );

	for (int i=1; i<__UI_FUNCTIONS;i++)
		if (!strcmp( str, uiFuncs[i].string )) return uiFuncs[i].uiID;

	return UI_UNKNOWN;
}


int SceneDefinedUI::MapStringToKey( char *str, bool verbose ) const
{
	if (!str)
	{
		if (verbose) printf("**** Error!  MapStringToKey() passed NULL string!\n");
		return 0;
	}

	MakeLower( str );

	int len = (int)strlen( str );
	if (len == 1) return str[0];

	for (int i=0;i<NUM_KEYSTRINGS;i++)
		if (!strcmp( str, keyStrings[i].string )) return keyStrings[i].uiID;

	if (verbose) printf("**** MapStringToKey() Error!  Unknown key '%s'.  Using '%c'.\n", str, str[0]);
	return str[0];
}



int SceneDefinedUI::ApplyKeyModifiers( int unmodifiedKey, int modifiers ) const
{
	return (unmodifiedKey & KEY_UNMODIFIED_MASK) | modifiers ;
}

bool SceneDefinedUI::IsModifier( int key ) const
{	
	return (key & KEY_MODIFIER_MASK) > 0;
}

int SceneDefinedUI::ConvertGLUTKey( unsigned char key, int modifiers ) const
{
	int theKey = (int)key;
	int ourModifiers = KEY_NO_MODIFIER;
	if (modifiers & GLUT_ACTIVE_SHIFT) ourModifiers |= KEY_MODIFIER_SHIFT;
	if (modifiers & GLUT_ACTIVE_CTRL)  ourModifiers |= KEY_MODIFIER_CONTROL;
	if (modifiers & GLUT_ACTIVE_ALT)   ourModifiers |= KEY_MODIFIER_ALT;

	if (key >= 'A' && key <= 'Z') theKey = (int)(key-'A'+'a');

	return (theKey & KEY_UNMODIFIED_MASK) | ourModifiers;
}

int SceneDefinedUI::ConvertGLUTModifiers( int modifiers ) const
{
	int ourModifiers = KEY_NO_MODIFIER;
	if (modifiers & GLUT_ACTIVE_SHIFT) ourModifiers |= KEY_MODIFIER_SHIFT;
	if (modifiers & GLUT_ACTIVE_CTRL)  ourModifiers |= KEY_MODIFIER_CONTROL;
	if (modifiers & GLUT_ACTIVE_ALT)   ourModifiers |= KEY_MODIFIER_ALT;
	return ourModifiers;
}

int SceneDefinedUI::ConvertGLUTTrackball( int button, int modifiers ) const
{
	int ourModifiers = KEY_NO_MODIFIER;
	if (modifiers & GLUT_ACTIVE_SHIFT) ourModifiers |= KEY_MODIFIER_SHIFT;
	if (modifiers & GLUT_ACTIVE_CTRL)  ourModifiers |= KEY_MODIFIER_CONTROL;
	if (modifiers & GLUT_ACTIVE_ALT)   ourModifiers |= KEY_MODIFIER_ALT;

	int base = MOUSE_LBUTTON;  
	if (button == GLUT_LEFT_BUTTON) base += 0;
	else if (button == GLUT_MIDDLE_BUTTON) base += 1;
	else if (button == GLUT_RIGHT_BUTTON) base += 2;

	int returnVal = (base & KEY_UNMODIFIED_MASK) | ourModifiers;

	for (int i = UI_EYE_TRACKBALL; i < UI_OBJECT_TRACKBALL_3; i++)
		if (keyMap[ i ] == returnVal) return returnVal;

	return KEY_UNKNOWN;
}

int SceneDefinedUI::ConvertGLUTMouseX( int button, int deltaX, int modifiers ) const
{
	int ourModifiers = KEY_NO_MODIFIER;
	if (modifiers & GLUT_ACTIVE_SHIFT) ourModifiers |= KEY_MODIFIER_SHIFT;
	if (modifiers & GLUT_ACTIVE_CTRL)  ourModifiers |= KEY_MODIFIER_CONTROL;
	if (modifiers & GLUT_ACTIVE_ALT)   ourModifiers |= KEY_MODIFIER_ALT;

	int base = MOUSE_BASE;  // +0 for X (+2 for Y)
	if (button == GLUT_LEFT_BUTTON) base += 4;
	else if (button == GLUT_MIDDLE_BUTTON) base += 8;
	else if (button == GLUT_RIGHT_BUTTON) base += 12;
	if (deltaX < 0) base++;

	return (base & KEY_UNMODIFIED_MASK) | ourModifiers;
}

int SceneDefinedUI::ConvertGLUTMouseY( int button, int deltaY, int modifiers ) const
{
	int ourModifiers = KEY_NO_MODIFIER;
	if (modifiers & GLUT_ACTIVE_SHIFT) ourModifiers |= KEY_MODIFIER_SHIFT;
	if (modifiers & GLUT_ACTIVE_CTRL)  ourModifiers |= KEY_MODIFIER_CONTROL;
	if (modifiers & GLUT_ACTIVE_ALT)   ourModifiers |= KEY_MODIFIER_ALT;

	int base = MOUSE_BASE + 2;  // +2 for Y (+0 for X)
	if (button == GLUT_LEFT_BUTTON) base += 4;
	else if (button == GLUT_MIDDLE_BUTTON) base += 8;
	else if (button == GLUT_RIGHT_BUTTON) base += 12;
	if (deltaY < 0) base++;

	return (base & KEY_UNMODIFIED_MASK) | ourModifiers;
}


int SceneDefinedUI::ConvertGLUTSpecialKey( int key, int modifiers ) const
{
	int theKey = KEY_UNKNOWN;
	int ourModifiers = KEY_NO_MODIFIER;
	if (modifiers & GLUT_ACTIVE_SHIFT) ourModifiers |= KEY_MODIFIER_SHIFT;
	if (modifiers & GLUT_ACTIVE_CTRL)  ourModifiers |= KEY_MODIFIER_CONTROL;
	if (modifiers & GLUT_ACTIVE_ALT)   ourModifiers |= KEY_MODIFIER_ALT;

	switch( key )
	{
	default:
		break;
	case GLUT_KEY_F1:
	case GLUT_KEY_F2:
	case GLUT_KEY_F3:
	case GLUT_KEY_F4:
	case GLUT_KEY_F5:
	case GLUT_KEY_F6:
	case GLUT_KEY_F7:
	case GLUT_KEY_F8:
	case GLUT_KEY_F9:
	case GLUT_KEY_F10:
	case GLUT_KEY_F11:
	case GLUT_KEY_F12:
		theKey = KEY_F1+key-1;
		break;
	case GLUT_KEY_LEFT:			theKey = KEY_LEFT_ARROW; break;
	case GLUT_KEY_UP:			theKey = KEY_UP_ARROW; break;
	case GLUT_KEY_RIGHT:		theKey = KEY_RIGHT_ARROW; break;
	case GLUT_KEY_DOWN:			theKey = KEY_DOWN_ARROW; break;
	case GLUT_KEY_PAGE_UP:		theKey = KEY_PAGE_UP; break;
	case GLUT_KEY_PAGE_DOWN:	theKey = KEY_PAGE_DOWN; break;
	case GLUT_KEY_INSERT:		theKey = KEY_INSERT; break;
	case GLUT_KEY_HOME:			theKey = KEY_HOME; break;
	case GLUT_KEY_END:			theKey = KEY_END; break;
	}

	return theKey != KEY_UNKNOWN ? ((theKey & KEY_UNMODIFIED_MASK) | ourModifiers) : KEY_UNKNOWN;
}


