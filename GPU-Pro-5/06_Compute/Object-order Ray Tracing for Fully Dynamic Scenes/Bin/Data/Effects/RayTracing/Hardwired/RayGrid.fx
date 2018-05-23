/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#define BE_VOXEL_REP_SETUP
#define BE_TRACING_SETUP

#include <Pipelines/Tracing/VoxelRep.fx>
#include <Pipelines/Tracing/Ray.fx>
#include <Pipelines/Tracing/RaySet.fx>
#include <Pipelines/Tracing/CompactRay.fx>

#include <Utility/Math.fx>
#include <Utility/Bits.fx>

#define ACTIVE_RAY_FILTERING // Filter out terminated rays before entering ray marching shader.
// #define DATA_PARALLEL // Alternative to persistent threads, needs to be enabled together with DATA_PARALLEL in RayGrid.cpp.
// #define RAY_LENGTH_DEBUG // Debug switch to count number of steps taken per ray. May be displayed via color coding in RayGen.fx.
// #define STEP_DIVERGENCE_DEBUG // Debug switch to count number of steps taken per THREAD. May be displayed via color coding in RayGen.fx.

// Rays
RWStructuredBuffer<RayDesc> RayQueueUAV : RayQueueUAV;
RWStructuredBuffer<TracedGeometry> RayGeometryUAV : RayGeometryUAV;
RWStructuredBuffer<RayDebug> RayDebugUAV : RayDebugUAV;

// Voxel approximation of the scene
Texture3D<uint> Voxels : Voxels;

// Output buffers for ray links = (cell id, ray id)
RWByteAddressBuffer RayGridIdxListUAV : RayGridIdxListUAV;
RWByteAddressBuffer RayLinkListUAV : RayLinkListUAV;
uint RayGridIdxBaseAddress : RayGridIdxBaseAddress;
uint RayLinkBaseAddress : RayLinkBaseAddress;

// Output counter
RWStructuredBuffer<uint> CounterUAV : CounterUAV;

#ifdef ACTIVE_RAY_FILTERING
	// IDs of the rays that are still active
	ByteAddressBuffer ActiveRayList : ActiveRayList;
#endif

// Number of occupied ray grid cells & layers to expand in this pass
uint4 RayExitCountLimitsMinMax : BoundedTraversalLimits;

// Pack 3-component cell indices into 32 bit keys
uint makeGridKey(uint3 gridIdx, uint order = 0) { return bitzip(gridIdx); } // Morton code, so sorting by key yields z-order layout
uint3 gridIdxFromKey(uint gridKey) { return bitunzip3(gridKey); }
bool gridKeysEqual(uint gridKey1, uint gridKey2) { return gridKey1 == gridKey2; }

static const uint ScatterGroupSize = 256;

[numthreads(ScatterGroupSize,1,1)]
void CSScatterRays(uint dispatchIdx : SV_DispatchThreadID, uint threadIdx : SV_GroupIndex, uniform bool bOpenRaysOnly)
{
#ifdef STEP_DIVERGENCE_DEBUG
	uint totalStepCount = 0;
#endif

	uint rayCount = 
#ifdef ACTIVE_RAY_FILTERING
			(bOpenRaysOnly) ? RaySet.ActiveRayCount : RaySet.RayCount;
#else
			RaySet.RayCount;
#endif

	uint rayOffset = 0;
	uint rayIdx = rayOffset + dispatchIdx;
	
	bool rayLinksAvailable = true;

#ifndef DATA_PARALLEL
	// Query first ray to process
	rayIdx = rayOffset + RayQueueUAV.IncrementCounter();

	// Keep running until all rays projected
	[allow_uav_condition]
	while (rayIdx < rayCount)
	{
#else
	if (rayIdx < rayCount)
	{
#endif
#ifdef RAY_LENGTH_DEBUG
		uint totalRayLength = 0;
#endif
		float startOffset = 0.0f;
		bool rayActive = true;

		if (bOpenRaysOnly)
		{
#ifdef ACTIVE_RAY_FILTERING
			rayIdx = ActiveRayList.Load(4 * rayIdx);
#else
			if (RayGeometryUAV[rayIdx].Depth != MaxRayDepth)
				rayActive = false;
#endif

#ifdef RAY_LENGTH_DEBUG
			totalRayLength = RayDebugUAV[rayIdx].RayLength;
#endif
			// Continue marching
			startOffset = asfloat(RayGeometryUAV[rayIdx].Normal.x);
		}

		float3 gridOrig;
		float3 gridDir;
		
		float3 step;

		uint3 gridIdx;
		int3 gridStep;
		float3 nextOffsets;
		float2 offsetCurrentNext;
		float rayEndOffset;

		bool continueMarching;

		// Fetch & transform next ray
		{
			RayDesc rayDesc = RayQueueUAV[rayIdx];

			gridOrig = (rayDesc.Orig - VoxelRep.Min) * VoxelRep.VoxelScale;
			gridDir = rayDesc.Dir * VoxelRep.VoxelScale;

			// Continue marching
			gridOrig += gridDir * startOffset;

			step = 1.0f / max(abs(gridDir), 1.0e-32f);
			float3 stepSign = sign1(gridDir);

			// Start marching
			gridIdx = (uint3) gridOrig;
			gridStep = (int3) (stepSign * 1.1f);
			nextOffsets = (saturate(stepSign) - stepSign * frac(gridOrig)) * step;
			offsetCurrentNext = float2(0.0f, min3(nextOffsets));

			rayEndOffset = min3( abs(saturate(stepSign) * float3(VoxelRep.Resolution.xyz) - gridOrig) * step );
			continueMarching = true;
		}

		bool occupied = true;
		bool overlapped = true;
		bool initialOverlap = true;
		bool inCountingLayer = false;

		uint overlappedCount = 0;
		uint exitCount = 0;

		int stepCount = 0;
		
		[allow_uav_condition]
		for (bool keepMarching = rayActive && rayLinksAvailable; keepMarching; )
		{
			if (!overlapped)
				initialOverlap = false;

			overlapped = occupied = false;

			uint voxelInfo = Voxels[gridIdx];
			overlapped = occupied = (voxelInfo != 0);

			// Count number of occupied voxels & connected occupied voxel layers traversed
			{
				bool wasInCountingLayer = inCountingLayer;
				// Layer == consecutive sequence of occupied voxels
				// The initial layer (containing the ray origin) is not counted
				inCountingLayer = inCountingLayer && overlapped || overlapped && !initialOverlap;

				// Number of overlapped occupied voxels
				overlappedCount += overlapped;
				// Number of connected occupied voxel layers traversed
				exitCount += wasInCountingLayer && !inCountingLayer;
			}

			// Inject ray into overlapped grid cells
			if (overlapped)
			{
				uint linkIdx = CounterUAV.IncrementCounter();
				if (linkIdx >= RaySet.MaxRayLinkCount)
				{
					rayLinksAvailable = false;
					// Break immediately, don't write out of bounds & don't continue stepping
					// -> Need to resume expansion right here in the next pass
					break;
				}

				RayLinkListUAV.Store(RayGridIdxBaseAddress + 4 * linkIdx, makeGridKey(gridIdx));
				RayLinkListUAV.Store(RayLinkBaseAddress + 4 * linkIdx, rayIdx);
			}

			// March along ray
			bool3 nextStepMask = (nextOffsets <= offsetCurrentNext.y);
			offsetCurrentNext.x = offsetCurrentNext.y;

			// Update cell idx & next offsets
			gridIdx += gridStep & -nextStepMask;
			nextOffsets += step * nextStepMask;

			offsetCurrentNext.y = min3(nextOffsets);

			// Check if still inside the grid
			continueMarching = offsetCurrentNext.x < rayEndOffset;
			++stepCount;

			// March until end of grid reached
			// or number of layers per ray in this pass exceeded
			// or max number of ray links per ray in this pass exceeded
			keepMarching = continueMarching && exitCount < RayExitCountLimitsMinMax.z && overlappedCount < RayExitCountLimitsMinMax.w; 
		}
		
		// Terminate rays that reached the end of the grid
		if (!continueMarching)
			RayGeometryUAV[rayIdx].Depth = MissedRayDepth;
		
		// Store the current offset in case expansion for this ray has to be continued in the next tracing pass
		if (rayActive)
			RayGeometryUAV[rayIdx].Normal.x = asuint(startOffset + offsetCurrentNext.x - 0.01f * min3(step));

		// Debug info
		{
#ifdef RAY_LENGTH_DEBUG
			totalRayLength += stepCount;
			RayDebugUAV[rayIdx].RayLength = totalRayLength;
#endif
#ifdef STEP_DIVERGENCE_DEBUG
			totalStepCount += stepCount;
#endif
		}

#ifndef DATA_PARALLEL
		// Query next ray to be processed
		rayIdx = rayOffset + RayQueueUAV.IncrementCounter();
#endif
	}

#ifdef STEP_DIVERGENCE_DEBUG
	RayDebugUAV[dispatchIdx].RayLength = totalStepCount;
#endif
}

RWByteAddressBuffer GroupDispatchUAV : GroupDispatchUAV;

[numthreads(1,1,1)]
void CSScatterRaysGroup(uniform bool bOpenRaysOnly)
{
	uint rayCount = (bOpenRaysOnly) ? RaySet.ActiveRayCount : RaySet.RayCount;

	GroupDispatchUAV.Store3(
			0,
			uint3( ceil_div(rayCount, ScatterGroupSize), 1, 1 )
		);
}

StructuredBuffer<TracedGeometry> RayGeometry : RayGeometry;
RWByteAddressBuffer ActiveRayListUAV : ActiveRayListUAV;

static const uint FilterRayGroupSize = 512;

[numthreads(FilterRayGroupSize,1,1)]
void CSFilterRays(uint dispatchIdx : SV_DispatchThreadID, uint threadIdx : SV_GroupIndex)
{
	uint rayIdx = dispatchIdx;

	// Extract active rays from global list of rays: MaxRayDepth indicates no hits so far
	if (rayIdx < RaySet.RayCount && RayGeometry[rayIdx].Depth == MaxRayDepth)
	{
		uint listIdx = CounterUAV.IncrementCounter();
		ActiveRayListUAV.Store(4 * listIdx, rayIdx);
	}
}

[numthreads(1,1,1)]
void CSFilterRayGroups()
{
	GroupDispatchUAV.Store3(
			0,
			uint3( ceil_div(RaySet.RayCount, FilterRayGroupSize), 1, 1 )
		);
}

Texture3D<uint> RayGridBegin : RayGridBegin;
Texture3D<uint> RayGridEnd : RayGridEnd;

RWTexture3D<uint> RayGridBeginUAV : RayGridBeginUAV;
RWTexture3D<uint> RayGridEndUAV : RayGridEndUAV;
ByteAddressBuffer RayGridIdxList : RayGridIdxList;

static const uint InjectGroupSize = 512;

[numthreads(InjectGroupSize,1,1)]
void CSInject(uint dispatchIdx : SV_DispatchThreadID, uint threadIdx : SV_GroupIndex)
{
	uint listIdx = dispatchIdx;

	if (listIdx >= RaySet.RayLinkCount)
		return;

	// Load cell keys of 2 consecutive ray links
	uint2 interleavedGridIdcs = RayGridIdxList.Load2(4 * listIdx);
	uint interleavedGridIdx = interleavedGridIdcs.x;
	uint nextInterleavedGridIdx = (listIdx + 1 != RaySet.RayLinkCount) ? interleavedGridIdcs.y : -1;

	// If cell keys differ, we have found the end of the previous cell
	// and the beginning of the next cell
	if (!gridKeysEqual(interleavedGridIdx, nextInterleavedGridIdx))
	{
		uint3 gridIdx = gridIdxFromKey(interleavedGridIdx);
		RayGridEndUAV[gridIdx] = listIdx + 1;

		uint3 nextGridIdx = gridIdxFromKey(nextInterleavedGridIdx);
		RayGridBeginUAV[nextGridIdx] = listIdx + 1;
	}
}

[numthreads(1,1,1)]
void CSInjectGroup()
{
	GroupDispatchUAV.Store3(
			0,
			uint3( ceil_div(RaySet.RayLinkCount, InjectGroupSize), 1, 1 )
		);
}

StructuredBuffer<RayDesc> RayQueue : RayQueue;

ByteAddressBuffer RayLinkList : RayLinkList;
RWByteAddressBuffer RayListUAV : RayListUAV;

static const uint CompactRayGroupSize = 512;

[numthreads(CompactRayGroupSize,1,1)]
void CSCompactRays(uint dispatchIdx : SV_DispatchThreadID, uint threadIdx : SV_GroupIndex)
{
	if (dispatchIdx >= RaySet.RayLinkCount)
		return;

	uint rayIdx = RayLinkList.Load(4 * dispatchIdx);
	RayDesc rayDesc = RayQueue[rayIdx];

	// Inline compacted version of the ray in parallel to the ray link array
	uint3 listNode;
	listNode.x = PackDirection(rayDesc.Dir);
	listNode.yz = PackCellOrigin(rayDesc.Orig);
	RayListUAV.Store3(12 * dispatchIdx, listNode);
}

[numthreads(1,1,1)]
void CSCompactRayGroups()
{
	GroupDispatchUAV.Store3(
			0,
			uint3( ceil_div(RaySet.RayLinkCount, CompactRayGroupSize), 1, 1 )
		);
}

technique11 March
{
	pass
	{
		SetComputeShader( CompileShader(cs_5_0, CSScatterRays(false)) );
	}
	pass
	{
		SetComputeShader( CompileShader(cs_5_0, CSScatterRays(true)) );
	}
	pass
	{
		SetComputeShader( CompileShader(cs_5_0, CSScatterRaysGroup(false)) );
	}
	pass
	{
		SetComputeShader( CompileShader(cs_5_0, CSScatterRaysGroup(true)) );
	}
}

technique11 Inject
{
	pass
	{
		SetComputeShader( CompileShader(cs_5_0, CSInject()) );
	}
	pass
	{
		SetComputeShader( CompileShader(cs_5_0, CSInjectGroup()) );
	}
}

technique11 CompactRays
{
	pass
	{
		SetComputeShader( CompileShader(cs_5_0, CSCompactRays()) );
	}
	pass
	{
		SetComputeShader( CompileShader(cs_5_0, CSCompactRayGroups()) );
	}
}

#ifdef ACTIVE_RAY_FILTERING
technique11 FilterRays
{
	pass
	{
		SetComputeShader( CompileShader(cs_5_0, CSFilterRays()) );
	}
	pass
	{
		SetComputeShader( CompileShader(cs_5_0, CSFilterRayGroups()) );
	}
}
#endif
