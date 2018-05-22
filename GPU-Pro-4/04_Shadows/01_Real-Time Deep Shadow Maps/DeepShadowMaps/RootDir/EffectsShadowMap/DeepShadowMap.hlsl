matrix WorldViewProj;
float Alpha;

struct LinkedListEntryDepthAlphaNext
{
	float depth;
	float alpha;
	int next;
};

#include "DeepShadowMapGlobal.hlsl"

RWStructuredBuffer<StartElementBufEntry> StartElementBuf;

RWStructuredBuffer<LinkedListEntryDepthAlphaNext> LinkedListBufDAN;
StructuredBuffer<LinkedListEntryDepthAlphaNext> LinkedListBufDANRO;

RWStructuredBuffer<LinkedListEntryWithPrev> LinkedListBufWP;
RWStructuredBuffer<LinkedListEntryNeighbors> NeighborsBuf;

struct VS_IN
{
	float4 pos : POSITION;
	float3 norm : NORMAL;
};

struct PS_IN
{
	float4 pos : SV_POSITION;
};

PS_IN vs_main(VS_IN input)
{
	PS_IN output = (PS_IN)0;
	output.pos = mul(input.pos, WorldViewProj);
	return output;
}

void ps_main(PS_IN input)
{	
	int counter = LinkedListBufDAN.IncrementCounter();
	LinkedListBufDAN[counter].depth = input.pos.z + 0.00002f;			
	LinkedListBufDAN[counter].alpha = Alpha;
	
	int originalVal;
	
	InterlockedExchange(StartElementBuf[((uint)input.pos.y) * Dimension + (uint)input.pos.x].start, counter, originalVal);
	LinkedListBufDAN[counter].next = originalVal;
}

struct LocalListEntry
{
	float depth;
	float alpha;
};

// QUICKSORT, not used
/*
void SortBuffer(inout LocalListEntry list[NUM_BUF_ELEMENTS], int m, int n)
{
   int i, j, k;
   float key;
   LocalListEntry temp;
   
   int mArr[NUM_BUF_ELEMENTS];
   int nArr[NUM_BUF_ELEMENTS];
   int numElems = 1;
   mArr[0] = m;
   nArr[0] = n;
 
	while(numElems > 0)
	{
		numElems--;
		m = mArr[numElems];
		n = nArr[numElems];
		if(m < n)
		{
			k = (m + n) / 2;
			temp = list[m];
			list[m] = list[k];
			list[k] = temp;
			key = list[m].depth;
			i = m + 1;
			j = n;
			while(i <= j)
			{
				while((i <= n) && (list[i].depth <= key))
					i++;
				while((j >= m) && (list[j].depth > key))
					j--;
				if( i < j)
				{
					temp = list[i];
					list[i] = list[j];
					list[j] = temp;
				}
			}

			temp = list[m];
			list[m] = list[j];
			list[j] = temp;

			// SortBuffer(list, m, j - 1);
			mArr[numElems] = m;
			nArr[numElems] = j - 1;
			numElems++;
			// SortBuffer(list, j + 1, n);
			mArr[numElems] = j + 1;
			nArr[numElems] = n;
			numElems++;
		}
	}
}
*/

[numthreads(16, 8, 1)]
void cs_sort(uint3 DTid : SV_DispatchThreadID)
{
	if(DTid.y > Dimension -1 || DTid.x > Dimension -1)
		return;
	LocalListEntry list[NUM_BUF_ELEMENTS];
	int nextPoints[NUM_BUF_ELEMENTS];
	
	int current;
	LocalListEntry temp;
	int start = StartElementBufRO[DTid.y * Dimension + DTid.x].start;
	if(start == -1)
		return;
		
	current = start;
	int numElems = 0;
	for(int i = 0; i < NUM_BUF_ELEMENTS; i++)
	{
		numElems++;
		list[i].depth = LinkedListBufDANRO[current].depth;
		list[i].alpha = LinkedListBufDANRO[current].alpha;
		nextPoints[i] = LinkedListBufDANRO[current].next;
		
		if(nextPoints[i] == -1)
			break;
			
		current = nextPoints[i];
	}
	nextPoints[NUM_BUF_ELEMENTS - 1] = -1;	// important: if all elements are used at one point - cut off
	
	// SortBuffer(list, 0, numElems - 1);
	// use insertion sort instead of quick sort
	int j;
	for(int i = 1; i < numElems; i++)
	{
		temp = list[i];
		j = i - 1;
		
		[loop]
		while(temp.depth < list[j].depth && j >= 0)
		{
			list[j+1] = list[j];
			j = j-1;
		}
		list[j+1] = temp;
	}
	
	// reduction
	float shadingBefore = 1.0f;
	for(int i = 0; i < numElems; i++)
	{
		float shadingCurrent = shadingBefore * (1.0f - list[i].alpha);
		
		if(shadingBefore - shadingCurrent > 0.001f)
		{
			shadingBefore = shadingCurrent;
			list[i].alpha = shadingCurrent;
		}
		else
		{
			numElems = i;
			nextPoints[i - 1] = -1;
			break;
		}
	}
	
	int prev = -1;
	current = start;
	for(int i = 0; i < numElems; i++)
	{
		LinkedListBufWP[current].depth = list[i].depth;
		LinkedListBufWP[current].next = nextPoints[i];
		LinkedListBufWP[current].shading = list[i].alpha;
		LinkedListBufWP[current].prev = prev;
		prev = current;
		current = nextPoints[i];
	}
}

[numthreads(16, 8, 1)]
void cs_link(uint3 DTid : SV_DispatchThreadID)
{
	if(DTid.y > Dimension -1 || DTid.x > Dimension -1)
		return;
	int currentCenter = StartElementBufRO[DTid.y * Dimension + DTid.x].start;
	
	if(currentCenter == -1)
		return;
	
	int currentRight, currentTop;
		
	if(DTid.x != Dimension - 1)
		currentRight  = StartElementBufRO[DTid.y * Dimension + DTid.x + 1].start;
	else
		currentRight  =  -1;
	
	if(DTid.y != 0)	
		currentTop    = StartElementBufRO[(DTid.y + 1) * Dimension + DTid.x].start;
	else
		currentTop    = -1;
	
	LinkedListEntryWithPrev currentEntryRight; 
	if(currentRight != -1)
		currentEntryRight = LinkedListBufWPRO[currentRight];
	LinkedListEntryWithPrev currentEntryTop;
	if(currentTop != -1)
		currentEntryTop = LinkedListBufWPRO[currentTop];

	LinkedListEntryWithPrev tempListEntry;
	
	float depth;
	for(int i = 0; i < NUM_BUF_ELEMENTS; i++)
	{
		depth = LinkedListBufWPRO[currentCenter].depth;
		
		for(int i = 0; i < NUM_BUF_ELEMENTS; i++)
		{
			if(currentRight == -1 || currentEntryRight.next == -1)
				break;
			tempListEntry = LinkedListBufWPRO[currentEntryRight.next];
			if(depth < tempListEntry.depth)
				break;
			currentRight = currentEntryRight.next;
			currentEntryRight = tempListEntry;
		}
		NeighborsBuf[currentCenter].right = currentRight;
		
		for(int i = 0; i < NUM_BUF_ELEMENTS; i++)
		{
			if(currentTop == -1 || currentEntryTop.next == -1)
				break;
			tempListEntry = LinkedListBufWPRO[currentEntryTop.next];
			if(depth < tempListEntry.depth)
				break;
			currentTop = currentEntryTop.next;
			currentEntryTop = tempListEntry;
		}
		NeighborsBuf[currentCenter].top = currentTop;
		
		currentCenter = LinkedListBufWPRO[currentCenter].next;
		if(currentCenter == -1)
			break;
	}
}

RasterizerState DisableCullingBias
{
    CullMode = NONE;
	SlopeScaledDepthBias = 25.0f;
};

technique11 Render
{
	pass P0
	{
		SetRasterizerState(DisableCullingBias);  
		SetVertexShader(CompileShader(vs_5_0, vs_main()));
		SetPixelShader (CompileShader(ps_5_0, ps_main()));
	}
}

technique11 Sort
{
	pass P0
	{
		SetComputeShader(CompileShader(cs_5_0, cs_sort()));
	}
}

technique11 Link
{
	pass P0
	{
		SetComputeShader(CompileShader(cs_5_0, cs_link()));
	}
}