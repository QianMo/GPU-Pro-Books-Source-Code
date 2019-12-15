#include "globals.shi"

RWStructuredBuffer<DecalInfo> decalInfoBuffer: register(u0, space0);

[numthreads(1, 1, 1)]
void main()
{    
  decalInfoBuffer[0].decalHitMask = 0xffffffff;
}
