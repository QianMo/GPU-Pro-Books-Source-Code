////////////////////////////////////////////////////
// Auto-generated code from the L-system description
////////////////////////////////////////////////////

#pragma once

#include "GrammarBasicTypes.h"

typedef unsigned int IDType;			// Type of Symbol ID-s

//*********************************************
// Struct of a module for the generation phase
//*********************************************

struct Module
{
	unsigned int symbolID;
	unsigned int colorID;
	unsigned int east;
	unsigned int isDoor;
	unsigned int north;
	D3DXVECTOR4 orientation;
	D3DXVECTOR4 position;
	float size;
	unsigned int south;
	unsigned int terminated;
	unsigned int west;

	Module( unsigned int symbolID_, unsigned int colorID_, unsigned int east_, unsigned int isDoor_, unsigned int north_, D3DXVECTOR4 orientation_, D3DXVECTOR4 position_, float size_, unsigned int south_, unsigned int terminated_, unsigned int west_ );
	Module();
	void setAttribute( const String& attributeName, const String& value );
	static String moduleTypeFingerprint() { return "colorIDuinteastuintisDooruintnorthuintorientationfloat4positionfloat4sizefloatsouthuintterminateduintwestuint"; }
};

//***********************************
// Sorted module, used for instancing
//***********************************

struct SortedModule
{
	unsigned int colorID;
	unsigned int east;
	unsigned int isDoor;
	unsigned int north;
	D3DXVECTOR4 orientation;
	D3DXVECTOR4 position;
	float size;
	unsigned int south;
	unsigned int terminated;
	unsigned int west;

	SortedModule( unsigned int colorID_, unsigned int east_, unsigned int isDoor_, unsigned int north_, D3DXVECTOR4 orientation_, D3DXVECTOR4 position_, float size_, unsigned int south_, unsigned int terminated_, unsigned int west_ );
	SortedModule();
};
