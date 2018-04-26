////////////////////////////////////////////////////
// Auto-generated code from the L-system description
////////////////////////////////////////////////////

#include "DXUT.h"
#include "Module.h"

#include <sstream>

//*****************
// Module functions
//*****************

Module::Module( unsigned int symbolID_, unsigned int colorID_, unsigned int east_, unsigned int isDoor_, unsigned int north_, D3DXVECTOR4 orientation_, D3DXVECTOR4 position_, float size_, unsigned int south_, unsigned int terminated_, unsigned int west_ ):symbolID(symbolID_),colorID(colorID_),east(east_),isDoor(isDoor_),north(north_),orientation(orientation_),position(position_),size(size_),south(south_),terminated(terminated_),west(west_)
{
}

Module::Module():symbolID(0),colorID(0),east(0),isDoor(0),north(0),orientation(D3DXVECTOR4(0.0F,0.0F,0.0F,0.0F)),position(D3DXVECTOR4(0.0F,0.0F,0.0F,0.0F)),size(0),south(0),terminated(0),west(0)
{
}

void Module::setAttribute( const String& attributeName, const String& value )
{
	std::istringstream sstr(value);
	if( attributeName == "symbolID" )
	{
		sstr >> symbolID;
	}
	if( attributeName == "colorID" )
	{
		sstr >> colorID;
	}
	if( attributeName == "east" )
	{
		sstr >> east;
	}
	if( attributeName == "isDoor" )
	{
		sstr >> isDoor;
	}
	if( attributeName == "north" )
	{
		sstr >> north;
	}
	if( attributeName == "orientation" )
	{
		sstr >> orientation.x;
		sstr >> orientation.y;
		sstr >> orientation.z;
		sstr >> orientation.w;
	}
	if( attributeName == "position" )
	{
		sstr >> position.x;
		sstr >> position.y;
		sstr >> position.z;
		position.w = 1.0F;
	}
	if( attributeName == "size" )
	{
		sstr >> size;
	}
	if( attributeName == "south" )
	{
		sstr >> south;
	}
	if( attributeName == "terminated" )
	{
		sstr >> terminated;
	}
	if( attributeName == "west" )
	{
		sstr >> west;
	}
}


//***********************
// SortedModule functions
//***********************

SortedModule::SortedModule( unsigned int colorID_, unsigned int east_, unsigned int isDoor_, unsigned int north_, D3DXVECTOR4 orientation_, D3DXVECTOR4 position_, float size_, unsigned int south_, unsigned int terminated_, unsigned int west_ ):colorID(colorID_),east(east_),isDoor(isDoor_),north(north_),orientation(orientation_),position(position_),size(size_),south(south_),terminated(terminated_),west(west_)
{
}

SortedModule::SortedModule():colorID(0),east(0),isDoor(0),north(0),orientation(D3DXVECTOR4(0.0F,0.0F,0.0F,0.0F)),position(D3DXVECTOR4(0.0F,0.0F,0.0F,0.0F)),size(0),south(0),terminated(0),west(0)
{
}

