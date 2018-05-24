#include "light_definitions.h"


Texture2D g_depth_tex : register( t0 );
StructuredBuffer<float3> g_vBoundsBuffer : register( t1 );
StructuredBuffer<SFiniteLightData> g_vLightData : register( t2 );


#define NR_THREADS			64

// output buffer
RWBuffer<uint4> g_vLightList : register( u0 );

#define MAX_NR_COARSE_ENTRIES		64
#define MAX_NR_PRUNED_ENTRIES		24

groupshared unsigned int coarseList[MAX_NR_COARSE_ENTRIES];
groupshared unsigned int prunedList[MAX_NR_COARSE_ENTRIES];		// temporarily support room for all 64 while in LDS

groupshared uint ldsZMin;
groupshared uint ldsZMax;
groupshared uint lightOffs;
#ifdef FINE_PRUNING_ENABLED
groupshared uint ldsDoesLightIntersect[2];
#endif
groupshared int ldsNrLightsFinal;


float GetLinearDepth(float3 vP)
{
	float4 v4Pres = mul(float4(vP,1.0), g_mInvScrProjection);
	return v4Pres.z / v4Pres.w;
}


float3 GetViewPosFromLinDepth(float2 v2ScrPos, float fLinDepth)
{
	float fSx = g_mScrProjection[0].x;
	float fCx = g_mScrProjection[2].x;
	float fSy = g_mScrProjection[1].y;
	float fCy = g_mScrProjection[2].y;

#ifdef LEFT_HAND_COORDINATES
	return fLinDepth*float3( ((v2ScrPos.x-fCx)/fSx), ((v2ScrPos.y-fCy)/fSy), 1.0 );
#else
	return fLinDepth*float3( -((v2ScrPos.x+fCx)/fSx), -((v2ScrPos.y+fCy)/fSy), 1.0 );
#endif
}


[numthreads(NR_THREADS, 1, 1)]
void main(uint threadID : SV_GroupIndex, uint3 u3GroupID : SV_GroupID)
{
	uint2 tileIDX = u3GroupID.xy;
	uint t=threadID;

	if(t<MAX_NR_COARSE_ENTRIES)
		prunedList[t]=0;
	
	
	uint iWidth;
	uint iHeight;
	g_depth_tex.GetDimensions(iWidth, iHeight);
	uint nrTilesX = (iWidth+15)/16;
	uint nrTilesY = (iHeight+15)/16;

	// build tile scr boundary
	const uint uFltMax = 0x7f7fffff;  // FLT_MAX as a uint
	if(t==0)
	{
		ldsZMin = uFltMax;
		ldsZMax = 0;
		lightOffs = 0;
	}

#if !defined(XBONE) && !defined(PLAYSTATION4)
	GroupMemoryBarrierWithGroupSync();
#endif


	uint2 viTilLL = 16*tileIDX;

	// establish min and max depth first
	float dpt_mi=asfloat(uFltMax), dpt_ma=0.0;

	for(int idx=t; idx<256; idx+=NR_THREADS)
	{
		const float fDpth = g_depth_tex.Load( uint3(viTilLL.x+(idx&0xf), viTilLL.y+(idx>>4), 0) ).x;
		if(fDpth<VIEWPORT_SCALE_Z)		// if not skydome
		{
			dpt_mi = min(fDpth, dpt_mi);
			dpt_ma = max(fDpth, dpt_ma);
		}
	}

	InterlockedMax(ldsZMax, asuint(dpt_ma) );
	InterlockedMin(ldsZMin, asuint(dpt_mi) );


#if !defined(XBONE) && !defined(PLAYSTATION4)
	GroupMemoryBarrierWithGroupSync();
#endif


	float3 vTileLL = float3(viTilLL.x/(float) iWidth, viTilLL.y/(float) iHeight, asfloat(ldsZMin));
	float3 vTileUR = float3((viTilLL.x+16)/(float) iWidth, (viTilLL.y+16)/(float) iHeight, asfloat(ldsZMax));
	

	// build coarse list using AABB
	for(int l=(int) t; l<(int) g_iNrVisibLights; l += NR_THREADS)
	{
		const float3 vMi = g_vBoundsBuffer[l];
		const float3 vMa = g_vBoundsBuffer[l+g_iNrVisibLights];

		if( all(vMa>vTileLL) && all(vMi<vTileUR))
		{
			unsigned int uInc = 1;
			unsigned int uIndex;
			InterlockedAdd(lightOffs, uInc, uIndex);
			if(uIndex<MAX_NR_COARSE_ENTRIES) coarseList[uIndex] = l;		// add to light list
		}
	}

#ifdef FINE_PRUNING_ENABLED	
	if(t<2) ldsDoesLightIntersect[t] = 0;
#endif

#if !defined(XBONE) && !defined(PLAYSTATION4)
	GroupMemoryBarrierWithGroupSync();
#endif

	int iNrCoarseLights = lightOffs<MAX_NR_COARSE_ENTRIES ? lightOffs : MAX_NR_COARSE_ENTRIES;


#ifndef FINE_PRUNING_ENABLED	
	{
		int iNrLightsOut = iNrCoarseLights<MAX_NR_PRUNED_ENTRIES ? iNrCoarseLights : MAX_NR_PRUNED_ENTRIES;
		if((int)t<iNrLightsOut) prunedList[t] = coarseList[t];
		if(t==0) ldsNrLightsFinal=iNrLightsOut;
	}
#else
	{
		float4 vLinDepths;
		for(int i=0; i<4; i++)
		{
			int idx = t + i*NR_THREADS;
			uint2 uCrd = uint2(viTilLL.x+(idx&0xf), viTilLL.y+(idx>>4));
			float3 v3ScrPos = float3(uCrd.x+0.5, uCrd.y+0.5, g_depth_tex.Load( uint3(uCrd.xy, 0) ).x);
			vLinDepths[i] = GetLinearDepth(v3ScrPos);
		}

		uint uLightsFlags[2] = {0,0};
		int l=0;
		// we need this outer loop for when we cannot assume a wavefront is 64 wide
		// since in this case we cannot assume the lights will remain sorted by type
#if !defined(XBONE) && !defined(PLAYSTATION4)
		while(l<iNrCoarseLights)
#endif
		{
			// fetch light
			int idxCoarse = l<iNrCoarseLights ? coarseList[l] : 0;
			uint uLgtType = l<iNrCoarseLights ? g_vLightData[idxCoarse].uLightType : 0;

			// spot and wedge lights
			while(l<iNrCoarseLights && (uLgtType==SPOT_CIRCULAR_LIGHT || uLgtType==WEDGE_LIGHT))
			{
				SFiniteLightData lgtDat = g_vLightData[idxCoarse];

				const float fSpotNearPlane = 0;		// don't have one right now

				// serially check 4 pixels
				uint uVal = 0;
				for(int i=0; i<4; i++)
				{
					int idx = t + i*NR_THREADS;
	
					float3 vVPos = GetViewPosFromLinDepth(uint2(viTilLL.x+(idx&0xf), viTilLL.y+(idx>>4)) + float2(0.5,0.5), vLinDepths[i]);
	
					// check pixel
					float3 fromLight = vVPos-lgtDat.vLpos.xyz;																					// nuts but vLdir is the X axis
					if(uLgtType==WEDGE_LIGHT) fromLight -= clamp(dot(fromLight, lgtDat.vLdir.xyz), 0, lgtDat.fSegLength) * lgtDat.vLdir.xyz;	// wedge light
					float distSq = dot(fromLight,fromLight);
					const float fProjVecMag = dot(fromLight, lgtDat.vBoxAxisX.xyz);

					if( all( float3(lgtDat.fSphRadiusSq, fProjVecMag, fProjVecMag) > float3(distSq, sqrt(distSq)*lgtDat.fPenumbra, fSpotNearPlane) ) ) uVal = 1;
				}

				uLightsFlags[l<32 ? 0 : 1] |= (uVal<<(l&31));
				++l; idxCoarse = l<iNrCoarseLights ? coarseList[l] : 0;
				uLgtType = l<iNrCoarseLights ? g_vLightData[idxCoarse].uLightType : 0;
			}

			// sphere and capsule test
			while(l<iNrCoarseLights && (uLgtType==SPHERE_LIGHT || uLgtType==CAPSULE_LIGHT))
			{
				SFiniteLightData lgtDat = g_vLightData[idxCoarse];

				// serially check 4 pixels
				uint uVal = 0;
				for(int i=0; i<4; i++)
				{
					int idx = t + i*NR_THREADS;
	
					float3 vVPos = GetViewPosFromLinDepth(uint2(viTilLL.x+(idx&0xf), viTilLL.y+(idx>>4)) + float2(0.5,0.5), vLinDepths[i]);
	
					// check pixel
					float3 vLp = lgtDat.vLpos.xyz;
					if(uLgtType==CAPSULE_LIGHT) vLp += clamp(dot(vVPos-vLp, lgtDat.vLdir.xyz), 0, lgtDat.fSegLength) * lgtDat.vLdir.xyz;	// wedge light
					float3 toLight = vLp - vVPos; 
					float distSq = dot(toLight,toLight);
			
					if(distSq<lgtDat.fSphRadiusSq) uVal = 1;
				}

				uLightsFlags[l<32 ? 0 : 1] |= (uVal<<(l&31));
				++l; idxCoarse = l<iNrCoarseLights ? coarseList[l] : 0;
				uLgtType = l<iNrCoarseLights ? g_vLightData[idxCoarse].uLightType : 0;
			}


			// box test
			while(l<iNrCoarseLights && uLgtType==BOX_LIGHT)
			{
				SFiniteLightData lgtDat = g_vLightData[idxCoarse];

				// serially check 4 pixels
				uint uVal = 0;
				for(int i=0; i<4; i++)
				{
					int idx = t + i*NR_THREADS;
	
					float3 vVPos = GetViewPosFromLinDepth(uint2(viTilLL.x+(idx&0xf), viTilLL.y+(idx>>4)) + float2(0.5,0.5), vLinDepths[i]);

					float3 toLight  = lgtDat.vLpos.xyz - vVPos;

					float3 dist = float3( dot(toLight, lgtDat.vBoxAxisX), dot(toLight, lgtDat.vLdir.xyz), dot(toLight, lgtDat.vBoxAxisZ) );
					dist = (abs(dist) - lgtDat.vBoxInnerDist) * lgtDat.vBoxInvRange;		// not as efficient as it could be
					if( max(max(dist.x, dist.y), dist.z)<1 ) uVal = 1;						// but allows us to not write out OuterDists
				}

				uLightsFlags[l<32 ? 0 : 1] |= (uVal<<(l&31));
				++l; idxCoarse = l<iNrCoarseLights ? coarseList[l] : 0;
				uLgtType = l<iNrCoarseLights ? g_vLightData[idxCoarse].uLightType : 0;
			}

#if !defined(XBONE) && !defined(PLAYSTATION4)
			// in case we have some corrupt data make sure we terminate
			if(uLgtType>=MAX_TYPES) ++l;
#endif
		}

		InterlockedOr(ldsDoesLightIntersect[0], uLightsFlags[0]);
		InterlockedOr(ldsDoesLightIntersect[1], uLightsFlags[1]);
		if(t==0) ldsNrLightsFinal = 0;

#if !defined(XBONE) && !defined(PLAYSTATION4)
		GroupMemoryBarrierWithGroupSync();
#endif

		if(t<iNrCoarseLights && (ldsDoesLightIntersect[t<32 ? 0 : 1]&(1<<(t&31)))!=0 )
		{
			unsigned int uInc = 1;
			unsigned int uIndex;
			InterlockedAdd(ldsNrLightsFinal, uInc, uIndex);
			if(uIndex<MAX_NR_COARSE_ENTRIES) prunedList[uIndex] = coarseList[t];		// we allow up to 64 pruned lights while stored in LDS.
		}
	}
#endif


#if !defined(XBONE) && !defined(PLAYSTATION4)
	GroupMemoryBarrierWithGroupSync();
#endif

	// no more than 24 pruned lights go out
	int nrLightsFinal = ldsNrLightsFinal<MAX_NR_PRUNED_ENTRIES ? ldsNrLightsFinal : MAX_NR_PRUNED_ENTRIES;

	int offs = tileIDX.y*nrTilesX + tileIDX.x;

	// write out lists in 10:10:10:2 format
	const uint uNumDWordsPerTile = MAX_NR_PRUNED_ENTRIES/3;		// 8
	if(t<uNumDWordsPerTile)
	{
		const int k = t*3;

		const int nrEntries = clamp(nrLightsFinal-k, 0, 3);
		g_vLightList[uNumDWordsPerTile*offs+t]=uint4(prunedList[k+0], prunedList[k+1], prunedList[k+2], nrEntries);		// write out uint4 as 10:10:10:2
	}

}