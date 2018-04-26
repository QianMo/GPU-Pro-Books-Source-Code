#ifndef __GR_MODULES_FX
#define __GR_MODULES_FX

#define __GR_POS_DEFINED

#define __GR_SIZE_DEFINED

#define __GR_ORIENTATION_DEFINED

#define __GR_TERMINATION_DEFINED

typedef uint IDType;
typedef float4 PositionType;


// Represents a module of the grammar and its parameters
struct Module
{
	uint symbolID : ID;
	uint colorID : COLORID;
	uint east : EAST;
	uint isDoor : ISDOOR;
	uint north : NORTH;
	float4 orientation : ORIENTATION;
	float4 position : POSITION;
	float size : SIZE;
	uint south : SOUTH;
	uint terminated : TERMINATED;
	uint west : WEST;
};

// Represents a module of the grammar after sorting and its parameters
struct sortedModule
{
	uint colorID : COLORID;
	uint east : EAST;
	uint isDoor : ISDOOR;
	uint north : NORTH;
	float4 orientation : ORIENTATION;
	float4 position : POSITION;
	float size : SIZE;
	uint south : SOUTH;
	uint terminated : TERMINATED;
	uint west : WEST;
};

// convert method from Module to SortedModule
sortedModule convertToSorted( Module input )
{
	sortedModule output;
	output.colorID = input.colorID;
	output.east = input.east;
	output.isDoor = input.isDoor;
	output.north = input.north;
	output.orientation = input.orientation;
	output.position = input.position;
	output.size = input.size;
	output.south = input.south;
	output.terminated = input.terminated;
	output.west = input.west;
	return output;
}

#define __GR_GENERATION_SO_ARG 	"ID.x; colorID.x; east.x; isDoor.x; north.x; orientation.xyzw; position.xyzw; size.x; south.x; terminated.x; west.x"
#define __GR_SORTING_SO_ARG 	"colorID.x; east.x; isDoor.x; north.x; orientation.xyzw; position.xyzw; size.x; south.x; terminated.x; west.x"

#endif
