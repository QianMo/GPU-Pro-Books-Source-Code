// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

//--------------------------------------------------------------------------------------------------------------------
// INTERSECTION STAGE
//--------------------------------------------------------------------------------------------------------------------
[numthreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1)]
void CSComputeIntersections(uint3 DTid : SV_DispatchThreadID)
{
	unsigned int index = DTid.y * N + DTid.x;

    if ( g_uRays[index].iTriangleId > (-2) )
	{
		// Uncomment a line to select an acceleration structure
		if(g_iAccelerationStructure == 0)
		{
			g_uIntersections[index] = BVH_IntersectP(g_uRays[index]);
		}
		/*else if(g_iAccelerationStructure == 9)
		{
    		g_uIntersections[index] = LBVH_IntersectP(g_uRays[index], g_uIntersections[index].iRoot);
		}*/
	}
	else
	{
		g_uIntersections[index].iTriangleId = -2;
	}
}