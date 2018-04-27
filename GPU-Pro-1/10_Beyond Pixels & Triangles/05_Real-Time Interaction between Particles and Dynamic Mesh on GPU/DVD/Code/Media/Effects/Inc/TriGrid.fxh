shared cbuffer def
{
	float rcpNumCells;
	float orient;
	float displacement;
	float dimmension;

	uint instanceID;
	uint mask;
};

#if 0
float isOutside( float3 diff )
{
	// these dots vs. conditions save ~5 instructions
	return dot( -min(diff, 0.), 1. ) + dot( max(diff - gridSize, 0.), 1. );
}
#endif
