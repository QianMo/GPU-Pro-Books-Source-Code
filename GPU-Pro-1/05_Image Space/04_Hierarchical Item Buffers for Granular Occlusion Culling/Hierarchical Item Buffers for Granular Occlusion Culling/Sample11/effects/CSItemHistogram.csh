RWTexture2DArray<uint> itemHistogram : register( u0 );
RWTexture2DArray<uint> itemHistogram0 : register( u1 );
Texture2D<uint> itemBuffer : register( t0 );


[numthreads(32,32,1)]
void CSMain( 
	uint3 GID  : SV_GROUPID,
	uint3 DTID : SV_DISPATCHTHREADID,
	uint3 GTID : SV_GROUPTHREADID,
	uint  GI   : SV_GROUPINDEX )
{
	// input dimensions
	uint w0,h0,s0;
	uint w1, h1;
	itemHistogram.GetDimensions( w0, h0, s0 );
	itemBuffer.GetDimensions( w1, h1 );
	
	// compute address
	uint iid		= itemBuffer[int2(DTID.xy)];
	uint arrayIndex	= iid;
	uint x			= ( DTID.x * w0 ) / w1;
	uint y			= ( DTID.y * h0 ) / h1;
	
	// interlocked increment
	uint oldVal;
	InterlockedAdd( itemHistogram[int3(x,y,iid)], 1, oldVal );
}