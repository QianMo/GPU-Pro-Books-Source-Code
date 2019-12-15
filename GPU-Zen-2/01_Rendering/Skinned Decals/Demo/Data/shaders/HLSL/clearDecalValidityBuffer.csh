#include "globals.shi"

RWStructuredBuffer<uint> decalValidityBuffer: register(u0, space0);

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
  decalValidityBuffer[dispatchThreadID.x] = 0;
}
