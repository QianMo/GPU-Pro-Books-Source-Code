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

// ------------------------------------------
// Checks if the ray hits the node BBox
// ------------------------------------------
float2 BVH_IntersectBox(float3 vfStart,float3 vfInvDir, unsigned int uiNodeNum) 
{	
	float2 T;

	float3 vfDiffMax = g_sNodes[uiNodeNum].vfMax-vfStart;
	vfDiffMax *= vfInvDir;
	float3 vfDiffMin = g_sNodes[uiNodeNum].vfMin-vfStart;
	vfDiffMin *= vfInvDir;

	T[0] = min(vfDiffMin.x,vfDiffMax.x);
	T[1] = max(vfDiffMin.x,vfDiffMax.x);

	T[0] = max(T[0],min(vfDiffMin.y,vfDiffMax.y));
	T[1] = min(T[1],max(vfDiffMin.y,vfDiffMax.y));

	T[0] = max(T[0],min(vfDiffMin.z,vfDiffMax.z));
	T[1] = min(T[1],max(vfDiffMin.z,vfDiffMax.z));

	//empty interval
	if (T[0] > T[1])
	{
		T[0] = T[1] = -1.0f;
	}

	return T;
}

// ------------------------------------------
// BVH intersection function
// ------------------------------------------
Intersection BVH_IntersectP(Ray ray)
{	
	// Initialize variables
	int iTrId = -1;
	float3 vfInvDir = float3(1.0f / ray.vfDirection);	
	
	int stack[35];
	int iStackOffset = 0;
	int iNodeNum = 0;
	int result = 0;
	
	// Initialize the current and best intersection.
	// They are empty at the beginning.
	Intersection cIntersection;
	Intersection bIntersection;
	bIntersection.iTriangleId = -1;
	bIntersection.fU = -1;
	bIntersection.fV = -1;
	bIntersection.fT = ray.fMaxT;
	bIntersection.iRoot = 0;
	bIntersection.iVisitedNodes = 0;
	
	[allow_uav_condition]
	while( true )
	{	
		// Perform ray-box intersection test
		float2 T = BVH_IntersectBox(ray.vfOrigin, vfInvDir, iNodeNum);
		result++;
		
		// If the ray does not intersect the box
		if ((T[0] > bIntersection.fT) || (T[1] < 0.0f))
		{
			// If the stack is empty, the traversal ends
			if(iStackOffset == 0) break;	
			// Pop a new node from the stack
			iNodeNum = stack[--iStackOffset];
		}
		// If the intersected box is a Leaf Node
		else if( g_sNodes[iNodeNum].nPrimitives > 0 )
		{
			[allow_uav_condition]
			for( int i = g_sNodes[iNodeNum].nPrimitives; i >= 0 ; --i )
			{
				result += 65536;
				// Get the triangle id contained by the node
				iTrId = g_sPrimitives[g_sNodes[iNodeNum].primitivesOffset];

				// Get the triangle data
				int offset = iTrId*3;
				float3 A = g_sVertices[g_sIndices[offset]].vfPosition.xyz;
				float3 B = g_sVertices[g_sIndices[offset+1]].vfPosition.xyz;
				float3 C = g_sVertices[g_sIndices[offset+2]].vfPosition.xyz;

				cIntersection = getIntersection(ray,A,B,C);
				// Search for an intersection:
				// 1. Avoid float-precision errors.
				// 2. Perform ray-triangle intersection test.
				// 3. Check if the new intersection is nearer to 
				// the camera than the current best intersection.
				if((ray.iTriangleId != iTrId)
					&& (RayTriangleTest(cIntersection)	)	
					&& (cIntersection.fT < bIntersection.fT))	
				{
					bIntersection = cIntersection;
					bIntersection.iTriangleId = iTrId;
					bIntersection.iRoot = iNodeNum;
				}
			}
			
			// If the stack is empty, the traversal ends
			if(iStackOffset == 0) break;					
			// Pop a new node from the stack
			iNodeNum = stack[--iStackOffset];
		}
		// If the intersected box is an Inner Node
		else
		{
			// Depending on the ray direction and the split-axis,
			// the order of the children changes on the stack.
			int dirIsNeg[3] = { vfInvDir < 0 };
			// -g_sNodes[iNodeNum].nPrimitives is the split axis: 0-1-2 (x-y-z)
			const int aux = dirIsNeg[-g_sNodes[iNodeNum].nPrimitives];
			// aux replaces an if/else statement which improves traversal a little bit
			stack[iStackOffset++] = (iNodeNum+1)*aux + (1-aux)*g_sNodes[iNodeNum].primitivesOffset;
			iNodeNum = g_sNodes[iNodeNum].primitivesOffset*aux + (1-aux)*(iNodeNum+1);
		}
	}

	bIntersection.iVisitedNodes = result;

	// return the best intersection found. If no intersection
	// was found, the intersection "contains" a triangle id = -1.
	return bIntersection;
}
