//-------------------------------------------------------------------------------------------------
// File: TerrainTessellation.fx
// Author: Ben Mistal
// Copyright 2010-2012 Mistal Research, Inc.
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// Shared #defines
//-------------------------------------------------------------------------------------------------
#define nNoiseSize					512
#define fIntToFloatConversionFactor	( 100.0f / 65536.0f )
#define nMinAppliedOffset			2

#define nMaxNumTerrainHeightOctaves	3
#define nMaxNumTerrainNormalOctaves	6
#define nNumOctaveWraps				2
#define fNoiseScale					0.0175f
#define fHeightScale				2.25f
#define fMinWeight					0.25f
#define fMinHeight					0.0f
#define fMaxHeight					4.0f

#define LeftSide	0
#define TopSide		1
#define RightSide	2
#define BottomSide	3

#define TopLeftVertex		0
#define TopRightVertex		1
#define BottomLeftVertex	2
#define BottomRightVertex	3

#define LeftClipPlaneMask	1
#define RightClipPlaneMask	2
#define TopClipPlaneMask	4
#define BottomClipPlaneMask	8
#define NearClipPlaneMask	16
#define FarClipPlaneMask	32
#define ClipPlaneMasks		( LeftClipPlaneMask | RightClipPlaneMask | TopClipPlaneMask | BottomClipPlaneMask | NearClipPlaneMask | FarClipPlaneMask )

#define AppliedOffsetIndex			3	
#define RelativeQuadrantCodeIndex	3

//-------------------------------------------------------------------------------------------------
// Structures
//-------------------------------------------------------------------------------------------------
struct FractalGeneratorInfoStruct
{
	float fOffset;
	float fGain;
	float fH;
	float fLacunarity;

}; // end struct FractalGeneratorInfoStruct

struct FractalOctaveInfoStruct
{
	float fSinArray[ 16 ];
	float fCosArray[ 16 ];
	float fReverseSinArray[ 16 ];
	float fReverseCosArray[ 16 ];
	float fXOffsetArray[ 16 ];
	float fYOffsetArray[ 16 ];
	float fExponentArray[ 16 ];

}; // end struct FractalGeneratorInfoStruct

struct GeneralFractalInfoStruct
{
	FractalGeneratorInfoStruct	fractalGeneratorInfo;
	FractalOctaveInfoStruct		fractalOctaveInfo;

}; // end struct GeneralFractalInfoStruct

struct RegionControlPointInfo
{
	float4 fvControlPoints[ 3 ][ 3 ];
	bool bNeighborRegionSplits[ 4 ];
	bool bParentNeighborRegionSplits[ 4 ];

}; // end struct RegionControlPointInfo

//-------------------------------------------------------------------------------------------------
// Shared Constants
//-------------------------------------------------------------------------------------------------
cbuffer constantBuffer : register( b0 )
{  
	float4x4 matWorld;
	float4x4 matView;
	float4x4 matProjection;
	float4x4 matWorldView;
	float4x4 matWorldViewProjection;
	float4   fvViewFrustumPlanes[ 6 ];
	float4   fvControlPosition;
	float4   fvEye;
	float4   fvLookAt;
	float4   fvUp;

	float	 fMaxRegionSpan;
	float	 fHSize;
	float	 fVSize;
	float	 fUnused;

	float4   fractalData[ 28 + 1 ]; // number of float4 (or equal) elements contained within GeneralFractalInfoStruct 

}; // end cbuffer constantBuffer : register( b0 )

cbuffer controlConstantBuffer : register( b1 )
{  
	float4 fvSplitControlPosition;

}; // end cbuffer controlConstantBuffer : register( b1 )

static GeneralFractalInfoStruct generalFractalInfo = ( GeneralFractalInfoStruct )fractalData; 

Texture2D		valueTexture2D			: register( t0 );
Texture2D		terrainPositionTexture	: register( t1 );

SamplerState	linearWrapSamplerState	: register( s0 );
SamplerState	pointClampSamplerState	: register( s1 );
SamplerState	pointWrapSamplerState	: register( s2 );

//-------------------------------------------------------------------------------------------------
// Forward Declarations
//-------------------------------------------------------------------------------------------------
struct VS_REGION_CONTROL_POINT_OUTPUT
{
    int4 RegionPosition	: POSITION0;	// Center_XYZ, AppliedOffset,
    int4 ParentInfo		: TEXCOORD0;	// Center_XYZ, RelativeQuadrantCode (same as vertex corner codes),

}; // end struct VS_REGION_CONTROL_POINT_OUTPUT

float  CalculateRegionSpan( const int3 nvPosition, const int nAppliedOffset );
float  CalculateRegionSpan( const float3 fvPosition, const int nAppliedOffset, const float fHeight );
float  CalculateRegionLerp( const float fRegionSpan );
float4 CalculateRegionLerps( const float4 fvRegionSpan );

bool   CanCullRegion( const int2 nvPosition, const int nAppliedOffset, const float fHeight );
bool   CanSplitRegion( const float fRegionSpan );

float  CalculateHeight( const float3 fvPosition, const uint nMaxNumOctaves );
float3 CalculateNormal(	const float3 fvPosition,
						const uint nMaxNumOctaves,
						const float fDistance );

float  DistanceFromPlane( const float3 fvPosition, const float4 fvPlaneEquation );
uint  GetFrustumCullMask( const float3 fvPosition );

RegionControlPointInfo CalculateRegionControlPointInfo( const VS_REGION_CONTROL_POINT_OUTPUT input );

//-------------------------------------------------------------------------------------------------
// Region Vertex Shader
//-------------------------------------------------------------------------------------------------
struct VS_REGION_CONTROL_POINT_INPUT
{
    int4 RegionPosition	: POSITION0;	// Center_XYZ, AppliedOffset,
    int4 ParentInfo		: TEXCOORD0;	// Center_XYZ, RelativeQuadrantCode (same as vertex corner codes),

}; // end struct VS_REGION_CONTROL_POINT_INPUT

VS_REGION_CONTROL_POINT_OUTPUT VS_Region_Pass_Through( VS_REGION_CONTROL_POINT_INPUT input )
{
	//---------------------------------------------------------------------
	// This vertex shader just passes through each region to the next stage
	//---------------------------------------------------------------------
    VS_REGION_CONTROL_POINT_OUTPUT output = ( VS_REGION_CONTROL_POINT_OUTPUT )0;

    output.RegionPosition	= input.RegionPosition;
    output.ParentInfo		= input.ParentInfo;

    return output;

} // end VS_REGION_CONTROL_POINT_OUTPUT VS_Region_Pass_Through( VS_REGION_CONTROL_POINT_INPUT input )

//-------------------------------------------------------------------------------------------------
// Composite Vertex Shader
//-------------------------------------------------------------------------------------------------
struct VS_COMPOSITE_INPUT
{
    float4 Position	: POSITION0;
	float4 Texcoord	: TEXCOORD0;

}; // end struct VS_COMPOSITE_INPUT

struct VS_COMPOSITE_OUTPUT
{
    float4 Position			: SV_POSITION;
	float4 Texcoord			: TEXCOORD0;
	float4 ScreenDirection	: TEXCOORD1; 

}; // end struct VS_COMPOSITE_OUTPUT

VS_COMPOSITE_OUTPUT VS_Composite( VS_COMPOSITE_INPUT input )
{
	//--------------------------------------------------------------------------------------
	// This vertex shader sets up the appropriate information to render the compositing quad
	//--------------------------------------------------------------------------------------
    VS_COMPOSITE_OUTPUT output = ( VS_COMPOSITE_OUTPUT )0;

    output.Position = input.Position;
    output.Texcoord = input.Texcoord;

	float fYRotation = radians( lerp( -30.0f, 30.0f, input.Texcoord.x ) );
	float fXRotation = radians( lerp( 30.0f, -30.0f, input.Texcoord.y ) );

	float4x4 matYRotation;
	matYRotation._11 = cos( fYRotation );
	matYRotation._21 = 0.0f;
	matYRotation._31 = -sin( fYRotation );
	matYRotation._41 = 0.0f;

	matYRotation._12 = 0.0f;
	matYRotation._22 = 1.0f;
	matYRotation._32 = 0.0f;
	matYRotation._42 = 0.0f;

	matYRotation._13 = sin( fYRotation );
	matYRotation._23 = 0.0f;
	matYRotation._33 = cos( fYRotation );
	matYRotation._43 = 0.0f;

	matYRotation._14 = 0.0f;
	matYRotation._24 = 0.0f;
	matYRotation._34 = 0.0f;
	matYRotation._44 = 1.0f;

	float4x4 matXRotation;
	matXRotation._11 = 1.0f;
	matXRotation._21 = 0.0f;
	matXRotation._31 = 0.0f;
	matXRotation._41 = 0.0f;

	matXRotation._12 = 0.0f;
	matXRotation._22 = cos( fXRotation );
	matXRotation._32 = sin( fXRotation );
	matXRotation._42 = 0.0f;

	matXRotation._13 = 0.0f;
	matXRotation._23 = -sin( fXRotation );
	matXRotation._33 = cos( fXRotation );
	matXRotation._43 = 0.0f;

	matXRotation._14 = 0.0f;
	matXRotation._24 = 0.0f;
	matXRotation._34 = 0.0f;
	matXRotation._44 = 1.0f;

	float3 fvScreenDirection = float3( 0.0f, 0.0f, 1.0f );
	float3 fvYRotation = mul( float4( fvScreenDirection, 0.0f ), matYRotation ).xyz;
	float3 fvXRotation = mul( float4( fvYRotation, 0.0f ), matXRotation ).xyz;
	output.ScreenDirection = mul( float4( fvXRotation, 0.0f ), transpose( matWorldView ) );

    return output;

} // end VS_COMPOSITE_OUTPUT VS_Composite( VS_COMPOSITE_INPUT input )

//-------------------------------------------------------------------------------------------------
// Split Region Geometry Shader
//-------------------------------------------------------------------------------------------------
struct GS_SPLIT_REGION_OUTPUT
{
    int4 RegionPosition	: POSITION0;	// Center_XYZ, AppliedOffset,
    int4 ParentInfo		: TEXCOORD0;	// Center_XYZ, RelativeQuadrantCode (same as vertex corner codes),

}; // end struct GS_SPLIT_REGION_OUTPUT

[ maxvertexcount( 4 ) ]
void GS_Split_Region(	point VS_REGION_CONTROL_POINT_OUTPUT input[ 1 ], 
						inout PointStream< GS_SPLIT_REGION_OUTPUT > intermediateRegionOutputStream,
						inout PointStream< GS_SPLIT_REGION_OUTPUT > finalRegionOutputStream )
{	
	//-------------------------------------------------------------------------------------
	// This geometry shader processes each input region, and splits the region if necessary
	//-------------------------------------------------------------------------------------
	if( !isfinite( input[ 0 ].RegionPosition[ AppliedOffsetIndex ] ) ) return; // This has been marked as invalid or unneeded
	if( input[ 0 ].RegionPosition[ AppliedOffsetIndex ] <= 0 ) return; // This has been marked as invalid or unneeded

	GS_SPLIT_REGION_OUTPUT output = ( GS_SPLIT_REGION_OUTPUT )0;

	int3 nvCenterPosition	= input[ 0 ].RegionPosition.xyz;
	int nAppliedOffset		= input[ 0 ].RegionPosition[ AppliedOffsetIndex ];

	float3 fvCenterPosition	= float3( float( nvCenterPosition.x ), 0.0f, float( nvCenterPosition.z ) ) * fIntToFloatConversionFactor;
	float  fCenterHeight	= CalculateHeight( fvCenterPosition, nMaxNumTerrainHeightOctaves );

	if( CanCullRegion( nvCenterPosition.xz, nAppliedOffset, fCenterHeight ) ) return;

	float fRegionSpan	= CalculateRegionSpan( fvCenterPosition, nAppliedOffset, fCenterHeight );
	bool bSplitRegion	= CanSplitRegion( fRegionSpan );

	if( fvSplitControlPosition.w >= 1.0f ) bSplitRegion = false; // This is the last loop through the splitting algorithm
	if( nAppliedOffset <= nMinAppliedOffset ) bSplitRegion = false; // This is the last loop through the splitting algorithm

	if( bSplitRegion )
	{
		//---------------------------------------
		// Split this region into four subregions
		//---------------------------------------
		int nSplitOffset = nAppliedOffset >> 1;

		output.ParentInfo.xyz = nvCenterPosition;

		output.ParentInfo[ RelativeQuadrantCodeIndex ]	= TopLeftVertex;
		output.RegionPosition							= int4( nvCenterPosition.x - nSplitOffset, 0, nvCenterPosition.z + nSplitOffset, nSplitOffset );
		intermediateRegionOutputStream.Append( output );

		output.ParentInfo[ RelativeQuadrantCodeIndex ]	= TopRightVertex;
		output.RegionPosition							= int4( nvCenterPosition.x + nSplitOffset, 0, nvCenterPosition.z + nSplitOffset, nSplitOffset );
		intermediateRegionOutputStream.Append( output );

		output.ParentInfo[ RelativeQuadrantCodeIndex ]	= BottomLeftVertex;
		output.RegionPosition							= int4( nvCenterPosition.x - nSplitOffset, 0, nvCenterPosition.z - nSplitOffset, nSplitOffset );
		intermediateRegionOutputStream.Append( output );

		output.ParentInfo[ RelativeQuadrantCodeIndex ]	= BottomRightVertex;
		output.RegionPosition							= int4( nvCenterPosition.x + nSplitOffset, 0, nvCenterPosition.z - nSplitOffset, nSplitOffset );
		intermediateRegionOutputStream.Append( output );

	} // end if( bSplitRegion )
	else
	{
		//-------------------------------------------
		// Pass this region through without splitting
		//-------------------------------------------
		output.RegionPosition	= input[ 0 ].RegionPosition;
		output.RegionPosition.y = asint( fCenterHeight );	
		output.ParentInfo		= input[ 0 ].ParentInfo;
		finalRegionOutputStream.Append( output );

	} // end else

} // end void GS_Split( ... )

//-------------------------------------------------------------------------------------------------
// Face / Wireframe Geometry Shader
//-------------------------------------------------------------------------------------------------
struct GS_TERRAIN_OUTPUT
{
	float4 Position	: SV_POSITION;
	float4 Color	: COLOR0;

}; // end struct GS_TERRAIN_OUTPUT

[ maxvertexcount( 2 * 2 * 2 * 2 ) ]
void GS_Region_Face(	point VS_REGION_CONTROL_POINT_OUTPUT input[ 1 ], 
						inout TriangleStream< GS_TERRAIN_OUTPUT > outputStream )
{	
	//-----------------------------------------------------------------------------------------------------
	// This geometry shader outputs the resulting faces from each input region, utilizing the LOD algorithm
	//-----------------------------------------------------------------------------------------------------
	if( input[ 0 ].RegionPosition[ AppliedOffsetIndex ] <= 0 ) return; // This has been marked as invalid or unneeded

	uint nXIndex;
	uint nYIndex;

	RegionControlPointInfo regionControlPointInfo = CalculateRegionControlPointInfo( input[ 0 ] );
	
	GS_TERRAIN_OUTPUT output = ( GS_TERRAIN_OUTPUT )0;
	output.Color = float4( 0.0f, 0.0f, 0.0f, 1.0f );

	{
		float4 fvProjectedControlPoints[ 3 ][ 3 ];
		fvProjectedControlPoints[ 0 ][ 0 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 0 ][ 0 ].xyz, 1.0f ), matWorldViewProjection );
		fvProjectedControlPoints[ 0 ][ 1 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 0 ][ 1 ].xyz, 1.0f ), matWorldViewProjection );
		fvProjectedControlPoints[ 0 ][ 2 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 0 ][ 2 ].xyz, 1.0f ), matWorldViewProjection );
		fvProjectedControlPoints[ 1 ][ 0 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 1 ][ 0 ].xyz, 1.0f ), matWorldViewProjection );
		fvProjectedControlPoints[ 1 ][ 1 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 1 ][ 1 ].xyz, 1.0f ), matWorldViewProjection );
		fvProjectedControlPoints[ 1 ][ 2 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 1 ][ 2 ].xyz, 1.0f ), matWorldViewProjection );
		fvProjectedControlPoints[ 2 ][ 0 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 2 ][ 0 ].xyz, 1.0f ), matWorldViewProjection );
		fvProjectedControlPoints[ 2 ][ 1 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 2 ][ 1 ].xyz, 1.0f ), matWorldViewProjection );
		fvProjectedControlPoints[ 2 ][ 2 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 2 ][ 2 ].xyz, 1.0f ), matWorldViewProjection );

		for( nXIndex = 0; nXIndex < 2; nXIndex++ )
		{
			for( nYIndex = 0; nYIndex < 3; nYIndex++ )
			{
				output.Position	= fvProjectedControlPoints[ nYIndex ][ nXIndex ];
				output.Color	= regionControlPointInfo.fvControlPoints[ nYIndex ][ nXIndex ];
				outputStream.Append( output );

				output.Position	= fvProjectedControlPoints[ nYIndex ][ nXIndex + 1 ];
				output.Color	= regionControlPointInfo.fvControlPoints[ nYIndex ][ nXIndex + 1 ];
				outputStream.Append( output );

			} // end for( nYIndex = 0; nYIndex < 3; nYIndex++ )

			outputStream.RestartStrip();

		} // end for( nXIndex = 0; nXIndex < 2; nXIndex++ )

	}

} // end void GS_Region_Face( ... )

[ maxvertexcount( 12 * 2 * 2 ) ]
void GS_Region_Wireframe(	point VS_REGION_CONTROL_POINT_OUTPUT input[ 1 ], 
							inout LineStream< GS_TERRAIN_OUTPUT > outputStream )
{	
	//---------------------------------------------------------------------------------------------------------
	// This geometry shader outputs the resulting wireframe from each input region, utilizing the LOD algorithm
	//---------------------------------------------------------------------------------------------------------
	if( input[ 0 ].RegionPosition[ AppliedOffsetIndex ] <= 0.0f ) return; // This has been marked as invalid or unneeded

	uint nXIndex;
	uint nYIndex;

	RegionControlPointInfo regionControlPointInfo = CalculateRegionControlPointInfo( input[ 0 ] );
	
	float4 fvProjectedControlPoints[ 3 ][ 3 ];
	fvProjectedControlPoints[ 0 ][ 0 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 0 ][ 0 ].xyz, 1.0f ), matWorldViewProjection ) + float4( 0.0f, 0.0f, 0.0f, 0.0001f );
	fvProjectedControlPoints[ 0 ][ 1 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 0 ][ 1 ].xyz, 1.0f ), matWorldViewProjection ) + float4( 0.0f, 0.0f, 0.0f, 0.0001f );
	fvProjectedControlPoints[ 0 ][ 2 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 0 ][ 2 ].xyz, 1.0f ), matWorldViewProjection ) + float4( 0.0f, 0.0f, 0.0f, 0.0001f );
	fvProjectedControlPoints[ 1 ][ 0 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 1 ][ 0 ].xyz, 1.0f ), matWorldViewProjection ) + float4( 0.0f, 0.0f, 0.0f, 0.0001f );
	fvProjectedControlPoints[ 1 ][ 1 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 1 ][ 1 ].xyz, 1.0f ), matWorldViewProjection ) + float4( 0.0f, 0.0f, 0.0f, 0.0001f );
	fvProjectedControlPoints[ 1 ][ 2 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 1 ][ 2 ].xyz, 1.0f ), matWorldViewProjection ) + float4( 0.0f, 0.0f, 0.0f, 0.0001f );
	fvProjectedControlPoints[ 2 ][ 0 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 2 ][ 0 ].xyz, 1.0f ), matWorldViewProjection ) + float4( 0.0f, 0.0f, 0.0f, 0.0001f );
	fvProjectedControlPoints[ 2 ][ 1 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 2 ][ 1 ].xyz, 1.0f ), matWorldViewProjection ) + float4( 0.0f, 0.0f, 0.0f, 0.0001f );
	fvProjectedControlPoints[ 2 ][ 2 ] = mul( float4( regionControlPointInfo.fvControlPoints[ 2 ][ 2 ].xyz, 1.0f ), matWorldViewProjection ) + float4( 0.0f, 0.0f, 0.0f, 0.0001f );

	GS_TERRAIN_OUTPUT output = ( GS_TERRAIN_OUTPUT )0;
	output.Color = float4( 1.0f, 1.0f, 1.0f, 1.0f );

	for( nYIndex = 0; nYIndex < 2; nYIndex++ )
	{
		for( nXIndex = 0; nXIndex < 2; nXIndex++ )
		{
			output.Position		= fvProjectedControlPoints[ nYIndex ][ nXIndex ];
			outputStream.Append( output );
			output.Position		= fvProjectedControlPoints[ nYIndex ][ nXIndex + 1 ];
			outputStream.Append( output );

			outputStream.RestartStrip();

			output.Position		= fvProjectedControlPoints[ nYIndex ][ nXIndex + 1 ];
			outputStream.Append( output );
			output.Position		= fvProjectedControlPoints[ nYIndex + 1 ][ nXIndex ];
			outputStream.Append( output );

			outputStream.RestartStrip();
			
			output.Position		= fvProjectedControlPoints[ nYIndex + 1 ][ nXIndex ];
			outputStream.Append( output );
			output.Position		= fvProjectedControlPoints[ nYIndex ][ nXIndex ];
			outputStream.Append( output );

			outputStream.RestartStrip();

			output.Position		= fvProjectedControlPoints[ nYIndex ][ nXIndex + 1 ];
			outputStream.Append( output );
			output.Position		= fvProjectedControlPoints[ nYIndex + 1 ][ nXIndex + 1 ];
			outputStream.Append( output );

			outputStream.RestartStrip();

			output.Position		= fvProjectedControlPoints[ nYIndex + 1 ][ nXIndex + 1 ];
			outputStream.Append( output );
			output.Position		= fvProjectedControlPoints[ nYIndex + 1 ][ nXIndex ];
			outputStream.Append( output );

			outputStream.RestartStrip();

			output.Position		= fvProjectedControlPoints[ nYIndex + 1 ][ nXIndex ];
			outputStream.Append( output );
			output.Position		= fvProjectedControlPoints[ nYIndex ][ nXIndex + 1 ];
			outputStream.Append( output );

			outputStream.RestartStrip();

		} // end for( nXIndex = 0; nXIndex < 2; nXIndex++ )

	} // end for( nYIndex = 0; nYIndex < 2; nYIndex++ )

} // end void GS_Wire( ... )

//-------------------------------------------------------------------------------------------------
// Region Pixel Shader
//-------------------------------------------------------------------------------------------------
struct PS_REGION_INPUT
{
	float4 Position	: SV_POSITION;
	float4 Color	: COLOR0;

}; // end struct PS_REGION_INPUT

struct PS_REGION_OUTPUT
{
    float4 Color	: SV_Target0;
 
}; // end struct PS_REGION_OUTPUT

PS_REGION_OUTPUT PS_Region( PS_REGION_INPUT input )
{
	//------------------------------------------------------------------------------
	// This pixel shader passes through the color, as set within the geometry shader
	//------------------------------------------------------------------------------
	PS_REGION_OUTPUT output = ( PS_REGION_OUTPUT )0;

	output.Color = input.Color;

	return output;

} // end PS_REGION_OUTPUT PS_Region( PS_REGION_INPUT input )

PS_REGION_OUTPUT PS_RegionWireframe( PS_REGION_INPUT input )
{
	//------------------------------------------------------------------------------
	// This pixel shader passes through the color, as set within the geometry shader
	//------------------------------------------------------------------------------
	PS_REGION_OUTPUT output = ( PS_REGION_OUTPUT )0;

	output.Color = input.Color;

	return output;

} // end PS_REGION_OUTPUT PS_RegionWireframe( PS_REGION_INPUT input )

//-------------------------------------------------------------------------------------------------
// Initialize / Composite Pixel Shader
//-------------------------------------------------------------------------------------------------
struct PS_COMPOSITE_INPUT
{
	float4 Position			: SV_POSITION;
	float2 Texcoord			: TEXCOORD0;
	float4 ScreenDirection	: TEXCOORD1; 

}; // end struct PS_COMPOSITE_INPUT

struct PS_COMPOSITE_OUTPUT
{
    float4 Color : SV_Target0;

}; // end struct PS_COMPOSITE_OUTPUT

float3 CalculateColor( float3 fvPosition, float fDistance, float fNDotL, float3 fvNormal )
{
	//-------------------------------------------------
	// Calculate the color of the terrain.  
	// White where the surface is deemed 'flat' enough,
	// and lerp between shades of grey where it is not.
	// Color is modulated with a diffuse lighting term. 
	// Colors also fade into the distance.
	//-------------------------------------------------
	float2 fvModifiedPosition = fvPosition.xz * 10.0f;
	float fModifiedNoise = valueTexture2D.Sample( linearWrapSamplerState, fvModifiedPosition ).w;

	float3 fvRockColor1 = float3( 0.4f, 0.3f, 0.25f );
	float3 fvRockColor2 = float3( 0.3f, 0.25f, 0.25f );
	float3 fvRockColor =lerp( fvRockColor1, fvRockColor2, fModifiedNoise );
	fvRockColor = fvRockColor * ( fNDotL * 0.5f + 0.25f );
	float3 fvColor = fvRockColor;

	float fSnowLerp = smoothstep( 0.6f, 0.85f, dot( fvNormal, float3( 0.0, 1.0, 0.0 ) ) );
	float3 fvSnowColor = float3( 224.0f, 224.0f, 255.0f ) / 255.0f;
	fvSnowColor = fvSnowColor * ( fNDotL * 0.65f + 0.35f );
	fvColor = lerp( fvRockColor, fvSnowColor, fSnowLerp );

	float fDistanceLerp = smoothstep( 0.0f, 35.0f, fDistance ) * 0.75f;
	float3 fvDistanceFadeColor = float3( 224.0f, 224.0f, 255.0f ) / 255.0f;
	fvColor = lerp( fvColor, fvDistanceFadeColor, fDistanceLerp );

	return fvColor;

} // end float3 CalculateColor( ... )

PS_COMPOSITE_OUTPUT PS_Background( PS_COMPOSITE_INPUT input )
{
	//------------------------
	// Calculate the sky color
	//------------------------
	PS_COMPOSITE_OUTPUT output = ( PS_COMPOSITE_OUTPUT )0;

	float3 fvSkyFadeColorBottom = float3( 224.0f, 224.0f, 255.0f ) / 255.0f;
	float3 fvSkyFadeColorTop = float3( 112.0f, 147.0f, 179.0f ) / 255.0f;
	float fSkyLerp = smoothstep( 0.0f, 0.625f, clamp( input.ScreenDirection.y + 0.15f, 0.0f, 1.0f ) );
	float3 fvSkyColor = lerp( fvSkyFadeColorBottom, fvSkyFadeColorTop, fSkyLerp );
	output.Color = float4( fvSkyColor, 1.0f );

	return output;

} // end PS_COMPOSITE_OUTPUT PS_Background( PS_COMPOSITE_INPUT input )

PS_COMPOSITE_OUTPUT PS_Composite( PS_COMPOSITE_INPUT input )
{
	//----------------------------
	// Calculate the terrain color
	//----------------------------
	PS_COMPOSITE_OUTPUT output = ( PS_COMPOSITE_OUTPUT )0;

	float4 fvTerrainPosition = terrainPositionTexture.Sample( pointClampSamplerState, float3( input.Texcoord.xy, 0.0f ) );

	float fDistance = fvTerrainPosition.w;

	float3 fvNormal = CalculateNormal(	fvTerrainPosition.xyz,
										nMaxNumTerrainNormalOctaves,
										fDistance );

	float3 fvLight = normalize( float3( 1.0f, 1.0f, -1.0f ) );
	float fNDotL = dot( fvNormal.xyz, fvLight );
	output.Color = float4( CalculateColor( fvTerrainPosition.xyz, fDistance, fNDotL, fvNormal.xyz ), 1.0f );

	return output;

} // end PS_COMPOSITE_OUTPUT PS_Composite( PS_COMPOSITE_INPUT input )

//-------------------------------------------------------------------------------------------------
// Helper Functions
//-------------------------------------------------------------------------------------------------
float CalculateRegionSpan( const int3 nvPosition, const int nAppliedOffset )
{
	float3 fvPosition	= float3( nvPosition ) * fIntToFloatConversionFactor;
	float  fHeight		= CalculateHeight( fvPosition, nMaxNumTerrainHeightOctaves );

	return CalculateRegionSpan( fvPosition, nAppliedOffset, fHeight );

} // end float CalculateRegionSpan( ... )

float CalculateRegionSpan( const float3 fvPosition, const int nAppliedOffset, const float fHeight )
{
	float  fAppliedOffset				= float( nAppliedOffset ) * fIntToFloatConversionFactor;
	float3 fvModifiedPosition			= float3( fvPosition.x, fHeight, fvPosition.z );
	float4 fvWorldPosition				= mul( float4( fvModifiedPosition, 1.0f), matWorldView );
	float4 fvLeftPosition				= mul( float4( fvWorldPosition.xyz, 1.0f ) - float4( fAppliedOffset, 0.0f, 0.0f, 0.0f ), matProjection );
	float4 fvRightPosition				= mul( float4( fvWorldPosition.xyz, 1.0f ) + float4( fAppliedOffset, 0.0f, 0.0f, 0.0f ), matProjection );
	float3 fvLeftNormalizedPosition		= ( fvLeftPosition.xyz / fvLeftPosition.w );
	float3 fvRightNormalizedPosition	= ( fvRightPosition.xyz / fvRightPosition.w );
	float  fRegionSpan					= length( fvRightNormalizedPosition.xy - fvLeftNormalizedPosition.xy ); 

	return fRegionSpan;

} // end float CalculateRegionSpan( ... )

bool CanSplitRegion( const float fRegionSpan )
{
	bool bSplit = false;
	
	if( fRegionSpan > fMaxRegionSpan ) bSplit = true;
	if( !isfinite( fRegionSpan ) ) bSplit = true;

	return bSplit;

} // end bool CanSplitRegion( ... )

float CalculateRegionLerp( const float fRegionSpan )
{
	return CalculateRegionLerps( ( float4 )fRegionSpan ).x;

} // end float CalculateRegionLerp( ... )

float4 CalculateRegionLerps( const float4 fvRegionSpans )
{
	float4 fvRegionLerps = clamp( ( fvRegionSpans / fMaxRegionSpan ) * 2.0f - 1.0f, 0.0f, 1.0f );
	
	return fvRegionLerps;

} // end float4 CalculateRegionLerps( ... )

float CalculateHeight( const float3 fvStartPosition, const uint nMaxNumOctaves )
{
	//-----------------------------
	// Calculate the terrain height
	//-----------------------------
	float2 fvPosition	= fvStartPosition.xz;
	float fHeight		= 0.0f;
	float fWeight		= 1.0f;

	for( uint nOctaveIndex = 0; nOctaveIndex < nMaxNumOctaves; nOctaveIndex++ )
	{
		float2 fvModifiedPosition = float2(	generalFractalInfo.fractalOctaveInfo.fCosArray[ nOctaveIndex ] * fvPosition.x + generalFractalInfo.fractalOctaveInfo.fSinArray[ nOctaveIndex ] * fvPosition.y,
											generalFractalInfo.fractalOctaveInfo.fCosArray[ nOctaveIndex ] * fvPosition.y - generalFractalInfo.fractalOctaveInfo.fSinArray[ nOctaveIndex ] * fvPosition.x );
		fvModifiedPosition += float2( generalFractalInfo.fractalOctaveInfo.fXOffsetArray[ nOctaveIndex ], generalFractalInfo.fractalOctaveInfo.fYOffsetArray[ nOctaveIndex ] );
		fvModifiedPosition = ( ( ( fvModifiedPosition * fNoiseScale ) % 1.0f ) + 1.0f ) % 1.0f;

		float4 fvSignals = valueTexture2D.SampleLevel( linearWrapSamplerState, fvModifiedPosition.xy, 0.0f );
		fvSignals.xyw *= fHeightScale;
		fvSignals.xyw *= fWeight;

		fHeight += fvSignals.w * generalFractalInfo.fractalOctaveInfo.fExponentArray[ nOctaveIndex ];

		fWeight = fvSignals.z * ( fvSignals.w / fHeightScale );
		fWeight *= generalFractalInfo.fractalGeneratorInfo.fGain;
		fWeight = clamp( fWeight, fMinWeight, 1.0 );

		fvPosition *= generalFractalInfo.fractalGeneratorInfo.fLacunarity;

	} // end for( uint nOctaveIndex = 0; nOctaveIndex < nMaxNumOctaves; nOctaveIndex++ )

	return fHeight;

} // end float CalculateHeight( ... );

float3 CalculateNormal( const float3 fvStartPosition, 
						const uint nMaxNumOctaves,
						const float fDistance )
{
	//--------------------------------------------------------
	// Calculate the terrain surface normal.
	// Adaptively adds noise octaves based on distance.
	// The noise texture we use contains gradient information,
	// which we use to 'build' the surface normal.
	//--------------------------------------------------------
	float2 fvPosition			= fvStartPosition.xz;
	float2 fvTotalDifferences	= float2( 0.0f, 0.0f );
	float  fWeight				= 1.0f;
	float  fOctaveWeight		= 1.0f;
	float  fOctaveLength		= fOctaveLength = ( ( 1.0f / ( float )nNoiseSize ) / fNoiseScale ) * generalFractalInfo.fractalGeneratorInfo.fLacunarity;

	float fFOV					= radians( 60.0f );
	float fOctaveCoefficient	= -log2( tan( ( fFOV / fHSize ) / 0.5f ) );
	float fNumOctaves			= ( fOctaveCoefficient - log2( fDistance ) + 0.5f ) / ( float )nNumOctaveWraps;
	uint nNumOctaves			= min( max( 1, ( uint )trunc( ceil( fNumOctaves ) ) ), nMaxNumOctaves );

	for( uint nOctaveIndex = 0; nOctaveIndex < nNumOctaves; nOctaveIndex++ )
	{
		if( ( float )nOctaveIndex >= ( fNumOctaves - 1.0f ) ) fOctaveWeight = frac( fNumOctaves );
		if( nOctaveIndex >= nNumOctaves ) break;

		float2 fvModifiedPosition = float2(	generalFractalInfo.fractalOctaveInfo.fCosArray[ nOctaveIndex ] * fvPosition.x + generalFractalInfo.fractalOctaveInfo.fSinArray[ nOctaveIndex ] * fvPosition.y,
											generalFractalInfo.fractalOctaveInfo.fCosArray[ nOctaveIndex ] * fvPosition.y - generalFractalInfo.fractalOctaveInfo.fSinArray[ nOctaveIndex ] * fvPosition.x );
		fvModifiedPosition += float2( generalFractalInfo.fractalOctaveInfo.fXOffsetArray[ nOctaveIndex ], generalFractalInfo.fractalOctaveInfo.fYOffsetArray[ nOctaveIndex ] );
		fvModifiedPosition = ( ( ( fvModifiedPosition * fNoiseScale ) % 1.0f ) + 1.0f ) % 1.0f;

		float4 fvSignals = valueTexture2D.SampleLevel( linearWrapSamplerState, fvModifiedPosition.xy, 0.0f );
		fvSignals.xyw *= fHeightScale;
		fvSignals.xyw *= fWeight;

		float2 fvLocalDifferences = ( fvSignals.xy - fvSignals.ww ) * generalFractalInfo.fractalOctaveInfo.fExponentArray[ nOctaveIndex ];
		float2 fvRotatedLocalDifferences = float2(	generalFractalInfo.fractalOctaveInfo.fReverseCosArray[ nOctaveIndex ] * fvLocalDifferences.x + generalFractalInfo.fractalOctaveInfo.fReverseSinArray[ nOctaveIndex ] * fvLocalDifferences.y,
													generalFractalInfo.fractalOctaveInfo.fReverseCosArray[ nOctaveIndex ] * fvLocalDifferences.y - generalFractalInfo.fractalOctaveInfo.fReverseSinArray[ nOctaveIndex ] * fvLocalDifferences.x );
		fvRotatedLocalDifferences *= fOctaveWeight;

		fvTotalDifferences /= generalFractalInfo.fractalGeneratorInfo.fLacunarity;
		fvTotalDifferences += fvRotatedLocalDifferences;

		fWeight = fvSignals.z * ( fvSignals.w / fHeightScale );
		fWeight *= generalFractalInfo.fractalGeneratorInfo.fGain;
		fWeight = clamp( fWeight, fMinWeight, 1.0 );

		fvPosition *= generalFractalInfo.fractalGeneratorInfo.fLacunarity;
		fOctaveLength /= generalFractalInfo.fractalGeneratorInfo.fLacunarity;

	} // end for( uint nOctaveIndex = 0; nOctaveIndex < nNumOctaves; nOctaveIndex++ )

	float3 fU		= normalize( float3( fOctaveLength, fvTotalDifferences.x, 0.0f ) );
	float3 fV		= normalize(float3( 0.0f, fvTotalDifferences.y, fOctaveLength ) );
	float3 fvNormal	= cross( fV, fU );

	float3 fvViewDirection = normalize( fvStartPosition.xyz - fvEye.xyz );
	if( dot( fvNormal, fvViewDirection ) > 0.0f )
	{
		fvNormal = normalize( fvNormal - dot( fvNormal, fvViewDirection ) * fvViewDirection );

	} // end if( dot( fvNormal, fvViewDirection ) >= 0.0f )
	else
	{
		fvNormal = normalize( fvNormal );

	} // end else

	return fvNormal;

} // end float3 CalculateNormal( ... );

float DistanceFromPlane( const float3 fvPosition, const float4 fvPlaneEquation )
{
    float fDistance = dot( float4( fvPosition, 1.0f ), fvPlaneEquation );
    
    return fDistance;

} // end float DistanceFromPlane( ... )

uint GetFrustumCullMask( const float3 fvPosition )
{
	uint nFrustumCullMask = 0;

    if( DistanceFromPlane( fvPosition, fvViewFrustumPlanes[ 0 ] ) <= 0.0f ) nFrustumCullMask |= LeftClipPlaneMask; // Left clip plane
    if( DistanceFromPlane( fvPosition, fvViewFrustumPlanes[ 1 ] ) <= 0.0f ) nFrustumCullMask |= RightClipPlaneMask; // Right clip plane
    if( DistanceFromPlane( fvPosition, fvViewFrustumPlanes[ 2 ] ) <= 0.0f ) nFrustumCullMask |= TopClipPlaneMask; // Top clip plane
    if( DistanceFromPlane( fvPosition, fvViewFrustumPlanes[ 3 ] ) <= 0.0f ) nFrustumCullMask |= BottomClipPlaneMask; // Bottom clip plane
    if( DistanceFromPlane( fvPosition, fvViewFrustumPlanes[ 4 ] ) <= 0.0f ) nFrustumCullMask |= NearClipPlaneMask; // Near clip plane
    if( DistanceFromPlane( fvPosition, fvViewFrustumPlanes[ 5 ] ) <= 0.0f ) nFrustumCullMask |= FarClipPlaneMask; // Far clip plane
        
    return nFrustumCullMask;

} // end uint GetFrustumCullMask( ... )

bool CanCullRegion( const int2 nvPosition, const int nAppliedOffset, const float fHeight )
{
	//----------------------------------------
	// Check if we can safely cull this region
	//----------------------------------------
	float3 fvPositions[ 4 ];

	fvPositions[ TopLeftVertex ]		= float3( float( nvPosition.x - nAppliedOffset ), 0.0f, float( nvPosition.y + nAppliedOffset ) ) * fIntToFloatConversionFactor;
	fvPositions[ TopRightVertex ]		= float3( float( nvPosition.x + nAppliedOffset ), 0.0f, float( nvPosition.y + nAppliedOffset ) ) * fIntToFloatConversionFactor;
	fvPositions[ BottomLeftVertex ]		= float3( float( nvPosition.x - nAppliedOffset ), 0.0f, float( nvPosition.y - nAppliedOffset ) ) * fIntToFloatConversionFactor;
	fvPositions[ BottomRightVertex ]	= float3( float( nvPosition.x + nAppliedOffset ), 0.0f, float( nvPosition.y - nAppliedOffset ) ) * fIntToFloatConversionFactor;

	uint nBottomFrustumCullMask = ClipPlaneMasks;
	uint nTopFrustumCullMask	= ClipPlaneMasks;

	float3 fvPosition		= float3( float( nvPosition.x ), 0.0f, float( nvPosition.y ) ) * fIntToFloatConversionFactor;
	float  fMaxSlope		= ( fMaxHeight - fMinHeight );
	float  fSlopeVariance	= fMaxSlope * float( nAppliedOffset ) * fIntToFloatConversionFactor;
	float  fLowerHeight		= fHeight - fSlopeVariance;
	float  fUpperHeight		= fHeight + fSlopeVariance;

	for( uint nIndex = 0; nIndex < 4; nIndex++ )
	{
		nBottomFrustumCullMask &= GetFrustumCullMask( fvPositions[ nIndex ] + float3( 0.0f, fLowerHeight, 0.0f ) );
		nTopFrustumCullMask &= GetFrustumCullMask( fvPositions[ nIndex ] + float3( 0.0f, fUpperHeight, 0.0f ) );
			
	} // end for( uint nIndex = 0; nIndex < 4; nIndex++ )

	return ( ( ( nBottomFrustumCullMask & nTopFrustumCullMask ) == 0 ) ? false : true );

} // end bool CanCullRegion( ... )

RegionControlPointInfo CalculateRegionControlPointInfo( const VS_REGION_CONTROL_POINT_OUTPUT input )
{
	//-----------------------------------------------------
	// Calculate all of the control points for this region.
	// Check the boundary cases to prevent any cracks.
	//-----------------------------------------------------
	RegionControlPointInfo regionControlPointInfo = ( RegionControlPointInfo )0;

	int3 nvCenterPosition	= input.RegionPosition.xyz;
	int  nAppliedOffset		= input.RegionPosition[ AppliedOffsetIndex ];

	float3 fvCenterPosition = float3( nvCenterPosition ) * fIntToFloatConversionFactor;
	float  fCenterHeight	= asfloat( input.RegionPosition.y );

	float3 fvCornerPositions[ 4 ];
	fvCornerPositions[ TopLeftVertex ]		= float3( float( nvCenterPosition.x - nAppliedOffset ) * fIntToFloatConversionFactor, fCenterHeight, float( nvCenterPosition.z + nAppliedOffset ) * fIntToFloatConversionFactor );
	fvCornerPositions[ TopRightVertex ]		= float3( float( nvCenterPosition.x + nAppliedOffset ) * fIntToFloatConversionFactor, fCenterHeight, float( nvCenterPosition.z + nAppliedOffset ) * fIntToFloatConversionFactor );
	fvCornerPositions[ BottomLeftVertex ]	= float3( float( nvCenterPosition.x - nAppliedOffset ) * fIntToFloatConversionFactor, fCenterHeight, float( nvCenterPosition.z - nAppliedOffset ) * fIntToFloatConversionFactor );
	fvCornerPositions[ BottomRightVertex ]	= float3( float( nvCenterPosition.x + nAppliedOffset ) * fIntToFloatConversionFactor, fCenterHeight, float( nvCenterPosition.z - nAppliedOffset ) * fIntToFloatConversionFactor );

	float4 fvCornerHeights;
	fvCornerHeights[ TopLeftVertex ]		= CalculateHeight( fvCornerPositions[ TopLeftVertex ], nMaxNumTerrainHeightOctaves );
	fvCornerHeights[ TopRightVertex ]		= CalculateHeight( fvCornerPositions[ TopRightVertex ], nMaxNumTerrainHeightOctaves );
	fvCornerHeights[ BottomLeftVertex ]		= CalculateHeight( fvCornerPositions[ BottomLeftVertex ], nMaxNumTerrainHeightOctaves );
	fvCornerHeights[ BottomRightVertex ]	= CalculateHeight( fvCornerPositions[ BottomRightVertex ], nMaxNumTerrainHeightOctaves );

	float3 fvMidpointPositions[ 4 ];
	fvMidpointPositions[ LeftSide ]		= float3( ( fvCornerPositions[ TopLeftVertex ].xyz + fvCornerPositions[ BottomLeftVertex ].xyz ) * 0.5f );
	fvMidpointPositions[ TopSide ]		= float3( ( fvCornerPositions[ TopLeftVertex ].xyz + fvCornerPositions[ TopRightVertex ].xyz ) * 0.5f );
	fvMidpointPositions[ RightSide ]	= float3( ( fvCornerPositions[ TopRightVertex ].xyz + fvCornerPositions[ BottomRightVertex ].xyz ) * 0.5f );
	fvMidpointPositions[ BottomSide ]	= float3( ( fvCornerPositions[ BottomLeftVertex ].xyz + fvCornerPositions[ BottomRightVertex ].xyz ) * 0.5f );

	float4 fvMidpointHeights;
	fvMidpointHeights[ LeftSide ]	= CalculateHeight( fvMidpointPositions[ LeftSide ], nMaxNumTerrainHeightOctaves );
	fvMidpointHeights[ TopSide ]	= CalculateHeight( fvMidpointPositions[ TopSide ], nMaxNumTerrainHeightOctaves );
	fvMidpointHeights[ RightSide ]	= CalculateHeight( fvMidpointPositions[ RightSide ], nMaxNumTerrainHeightOctaves );
	fvMidpointHeights[ BottomSide ]	= CalculateHeight( fvMidpointPositions[ BottomSide ], nMaxNumTerrainHeightOctaves );

	//------------------------------------------
	// Set the initial control point information
	//------------------------------------------
	regionControlPointInfo.fvControlPoints[ 0 ][ 0 ].xyz	= float3(	float( nvCenterPosition.x - nAppliedOffset ) * fIntToFloatConversionFactor,
																		fvCornerHeights[ TopLeftVertex ],
																		float( nvCenterPosition.z + nAppliedOffset ) * fIntToFloatConversionFactor );

	regionControlPointInfo.fvControlPoints[ 0 ][ 1 ].xyz	= float3(	float( nvCenterPosition.x ) * fIntToFloatConversionFactor,
																		fvMidpointHeights[ TopSide ],
																		float( nvCenterPosition.z + nAppliedOffset ) * fIntToFloatConversionFactor );

	regionControlPointInfo.fvControlPoints[ 0 ][ 2 ].xyz	= float3(	float( nvCenterPosition.x + nAppliedOffset ) * fIntToFloatConversionFactor,
																		fvCornerHeights[ TopRightVertex ],
																		float( nvCenterPosition.z + nAppliedOffset ) * fIntToFloatConversionFactor );

	regionControlPointInfo.fvControlPoints[ 1 ][ 0 ].xyz	= float3(	float( nvCenterPosition.x - nAppliedOffset ) * fIntToFloatConversionFactor,
																		fvMidpointHeights[ LeftSide ],
																		float( nvCenterPosition.z ) * fIntToFloatConversionFactor );

	regionControlPointInfo.fvControlPoints[ 1 ][ 1 ].xyz	= float3(	float( nvCenterPosition.x ) * fIntToFloatConversionFactor,
																		fCenterHeight,
																		float( nvCenterPosition.z ) * fIntToFloatConversionFactor );

	regionControlPointInfo.fvControlPoints[ 1 ][ 2 ].xyz	= float3(	float( nvCenterPosition.x + nAppliedOffset ) * fIntToFloatConversionFactor,
																		fvMidpointHeights[ RightSide ],
																		float( nvCenterPosition.z ) * fIntToFloatConversionFactor );

	regionControlPointInfo.fvControlPoints[ 2 ][ 0 ].xyz	= float3(	float( nvCenterPosition.x - nAppliedOffset ) * fIntToFloatConversionFactor,
																		fvCornerHeights[ BottomLeftVertex ],
																		float( nvCenterPosition.z - nAppliedOffset ) * fIntToFloatConversionFactor );

	regionControlPointInfo.fvControlPoints[ 2 ][ 1 ].xyz	= float3(	float( nvCenterPosition.x ) * fIntToFloatConversionFactor,
																		fvMidpointHeights[ BottomSide ],
																		float( nvCenterPosition.z - nAppliedOffset ) * fIntToFloatConversionFactor );

	regionControlPointInfo.fvControlPoints[ 2 ][ 2 ].xyz	= float3(	float( nvCenterPosition.x + nAppliedOffset ) * fIntToFloatConversionFactor,
																		fvCornerHeights[ BottomRightVertex ],
																		float( nvCenterPosition.z - nAppliedOffset ) * fIntToFloatConversionFactor );

	regionControlPointInfo.fvControlPoints[ 0 ][ 0 ].w = length( regionControlPointInfo.fvControlPoints[ 0 ][ 0 ].xyz - fvEye.xyz );
	regionControlPointInfo.fvControlPoints[ 0 ][ 1 ].w = length( regionControlPointInfo.fvControlPoints[ 0 ][ 1 ].xyz - fvEye.xyz );
	regionControlPointInfo.fvControlPoints[ 0 ][ 2 ].w = length( regionControlPointInfo.fvControlPoints[ 0 ][ 2 ].xyz - fvEye.xyz );
	regionControlPointInfo.fvControlPoints[ 1 ][ 0 ].w = length( regionControlPointInfo.fvControlPoints[ 1 ][ 0 ].xyz - fvEye.xyz );
	regionControlPointInfo.fvControlPoints[ 1 ][ 1 ].w = length( regionControlPointInfo.fvControlPoints[ 1 ][ 1 ].xyz - fvEye.xyz );
	regionControlPointInfo.fvControlPoints[ 1 ][ 2 ].w = length( regionControlPointInfo.fvControlPoints[ 1 ][ 2 ].xyz - fvEye.xyz );
	regionControlPointInfo.fvControlPoints[ 2 ][ 0 ].w = length( regionControlPointInfo.fvControlPoints[ 2 ][ 0 ].xyz - fvEye.xyz );
	regionControlPointInfo.fvControlPoints[ 2 ][ 1 ].w = length( regionControlPointInfo.fvControlPoints[ 2 ][ 1 ].xyz - fvEye.xyz );
	regionControlPointInfo.fvControlPoints[ 2 ][ 2 ].w = length( regionControlPointInfo.fvControlPoints[ 2 ][ 2 ].xyz - fvEye.xyz );

	//-------------------------------------------------------------------------------------
	// We need to calculate if the neighboring viewable regions have split.
	// This is to figure out if we are at an LOD border, and need to clamp our lerp values.
	//-------------------------------------------------------------------------------------
	int nNeighborOffset = nAppliedOffset << 1;

	float4 fvNeighborRegionSpans;
	fvNeighborRegionSpans[ LeftSide ]	= CalculateRegionSpan( nvCenterPosition + int3( -nNeighborOffset, 0, 0 ), nAppliedOffset );
	fvNeighborRegionSpans[ TopSide ]	= CalculateRegionSpan( nvCenterPosition + int3( 0, 0, nNeighborOffset ), nAppliedOffset );
	fvNeighborRegionSpans[ RightSide ]	= CalculateRegionSpan( nvCenterPosition + int3( nNeighborOffset, 0, 0 ), nAppliedOffset );
	fvNeighborRegionSpans[ BottomSide ]	= CalculateRegionSpan( nvCenterPosition + int3( 0, 0, -nNeighborOffset ), nAppliedOffset );

	bool bNeighborRegionSplits[ 4 ];
	bNeighborRegionSplits[ LeftSide ]	= ( nAppliedOffset > nMinAppliedOffset ) ? CanSplitRegion( fvNeighborRegionSpans[ LeftSide ] ) : false;
	bNeighborRegionSplits[ TopSide ]	= ( nAppliedOffset > nMinAppliedOffset ) ? CanSplitRegion( fvNeighborRegionSpans[ TopSide ] ) : false;
	bNeighborRegionSplits[ RightSide ]	= ( nAppliedOffset > nMinAppliedOffset ) ? CanSplitRegion( fvNeighborRegionSpans[ RightSide ] ) : false;
	bNeighborRegionSplits[ BottomSide ]	= ( nAppliedOffset > nMinAppliedOffset ) ? CanSplitRegion( fvNeighborRegionSpans[ BottomSide ] ) : false;

	regionControlPointInfo.bNeighborRegionSplits[ LeftSide ]	= bNeighborRegionSplits[ LeftSide ];
	regionControlPointInfo.bNeighborRegionSplits[ TopSide ]		= bNeighborRegionSplits[ TopSide ];
	regionControlPointInfo.bNeighborRegionSplits[ RightSide ]	= bNeighborRegionSplits[ RightSide ];
	regionControlPointInfo.bNeighborRegionSplits[ BottomSide ]	= bNeighborRegionSplits[ BottomSide ];

	fvNeighborRegionSpans[ LeftSide ]	= CalculateRegionSpan( nvCenterPosition + int3( -nAppliedOffset, 0, 0 ), nAppliedOffset );
	fvNeighborRegionSpans[ TopSide ]	= CalculateRegionSpan( nvCenterPosition + int3( 0, 0, nAppliedOffset ), nAppliedOffset );
	fvNeighborRegionSpans[ RightSide ]	= CalculateRegionSpan( nvCenterPosition + int3( nAppliedOffset, 0, 0 ), nAppliedOffset );
	fvNeighborRegionSpans[ BottomSide ]	= CalculateRegionSpan( nvCenterPosition + int3( 0, 0, -nAppliedOffset ), nAppliedOffset );

	float4 fvNeighborRegionLerps = CalculateRegionLerps( fvNeighborRegionSpans );

	if( bNeighborRegionSplits[ LeftSide ] )		fvNeighborRegionLerps[ LeftSide ] = 1.0f;
	if( bNeighborRegionSplits[ TopSide ] )		fvNeighborRegionLerps[ TopSide ] = 1.0f;
	if( bNeighborRegionSplits[ RightSide ] )	fvNeighborRegionLerps[ RightSide ] = 1.0f;
	if( bNeighborRegionSplits[ BottomSide ] )	fvNeighborRegionLerps[ BottomSide ] = 1.0f;

	float fRegionSpan	= CalculateRegionSpan( nvCenterPosition, nAppliedOffset );
	float fRegionLerp	= CalculateRegionLerp( fRegionSpan );

	//-------------------------------------------------------------------------------------
	// We need to calculate if the neighboring parent viewable regions have split.
	// This is to figure out if we are at an LOD border, and need to clamp our lerp values.
	//-------------------------------------------------------------------------------------
	int3 nvParentCenterPosition	= input.ParentInfo.xyz;
	int  nParentOffset			= nAppliedOffset << 1;
	int  nParentNeighborOffset	= nParentOffset << 1;

	float4 fvParentNeighborRegionSpans;
	fvParentNeighborRegionSpans[ LeftSide ]		= CalculateRegionSpan( nvParentCenterPosition + int3( -nParentNeighborOffset, 0, 0 ), nParentOffset );
	fvParentNeighborRegionSpans[ TopSide ]		= CalculateRegionSpan( nvParentCenterPosition + int3( 0, 0, nParentNeighborOffset ), nParentOffset );
	fvParentNeighborRegionSpans[ RightSide ]	= CalculateRegionSpan( nvParentCenterPosition + int3( nParentNeighborOffset, 0, 0 ), nParentOffset );
	fvParentNeighborRegionSpans[ BottomSide ]	= CalculateRegionSpan( nvParentCenterPosition + int3( 0, 0, -nParentNeighborOffset ), nParentOffset );

	bool bParentNeighborRegionSplits[ 4 ];
	bParentNeighborRegionSplits[ LeftSide ]		= CanSplitRegion( fvParentNeighborRegionSpans[ LeftSide ] );
	bParentNeighborRegionSplits[ TopSide ]		= CanSplitRegion( fvParentNeighborRegionSpans[ TopSide ] );
	bParentNeighborRegionSplits[ RightSide ]	= CanSplitRegion( fvParentNeighborRegionSpans[ RightSide ] );
	bParentNeighborRegionSplits[ BottomSide ]	= CanSplitRegion( fvParentNeighborRegionSpans[ BottomSide ] );

	regionControlPointInfo.bParentNeighborRegionSplits[ LeftSide ]		= bParentNeighborRegionSplits[ LeftSide ];
	regionControlPointInfo.bParentNeighborRegionSplits[ TopSide ]		= bParentNeighborRegionSplits[ TopSide ];
	regionControlPointInfo.bParentNeighborRegionSplits[ RightSide ]		= bParentNeighborRegionSplits[ RightSide ];
	regionControlPointInfo.bParentNeighborRegionSplits[ BottomSide ]	= bParentNeighborRegionSplits[ BottomSide ];

	if(    ( !bParentNeighborRegionSplits[ LeftSide ] )
	    && (    ( input.ParentInfo[ RelativeQuadrantCodeIndex ] == TopLeftVertex )
		     || ( input.ParentInfo[ RelativeQuadrantCodeIndex ] == BottomLeftVertex ) ) )
	{
		fvNeighborRegionLerps[ LeftSide ] = 0.0f;

	} // end if( ... )

	if(    ( !bParentNeighborRegionSplits[ TopSide ] )
	    && (    ( input.ParentInfo[ RelativeQuadrantCodeIndex ] == TopLeftVertex )
		     || ( input.ParentInfo[ RelativeQuadrantCodeIndex ] == TopRightVertex ) ) )
	{
		fvNeighborRegionLerps[ TopSide ] = 0.0f;

	} // end if( ... )

	if(    ( !bParentNeighborRegionSplits[ RightSide ] )
	    && (    ( input.ParentInfo[ RelativeQuadrantCodeIndex ] == TopRightVertex )
		     || ( input.ParentInfo[ RelativeQuadrantCodeIndex ] == BottomRightVertex ) ) )
	{
		fvNeighborRegionLerps[ RightSide ] = 0.0f;

	} // end if( ... )

	if(    ( !bParentNeighborRegionSplits[ BottomSide ] )
	    && (    ( input.ParentInfo[ RelativeQuadrantCodeIndex ] == BottomLeftVertex )
		     || ( input.ParentInfo[ RelativeQuadrantCodeIndex ] == BottomRightVertex ) ) )
	{
		fvNeighborRegionLerps[ BottomSide ] = 0.0f;

	} // end if( ... )

	//------------------------
	// Lerp our control points
	//------------------------ 
	if( !bNeighborRegionSplits[ LeftSide ] )	regionControlPointInfo.fvControlPoints[ 1 ][ 0 ] = lerp( regionControlPointInfo.fvControlPoints[ 0 ][ 0 ], regionControlPointInfo.fvControlPoints[ 1 ][ 0 ], fvNeighborRegionLerps[ LeftSide ] );
	if( !bNeighborRegionSplits[ TopSide ] )		regionControlPointInfo.fvControlPoints[ 0 ][ 1 ] = lerp( regionControlPointInfo.fvControlPoints[ 0 ][ 0 ], regionControlPointInfo.fvControlPoints[ 0 ][ 1 ], fvNeighborRegionLerps[ TopSide ] );
	regionControlPointInfo.fvControlPoints[ 1 ][ 1 ] = lerp( regionControlPointInfo.fvControlPoints[ 0 ][ 0 ], regionControlPointInfo.fvControlPoints[ 1 ][ 1 ], fRegionLerp );
	if( !bNeighborRegionSplits[ RightSide ] )	regionControlPointInfo.fvControlPoints[ 1 ][ 2 ] = lerp( regionControlPointInfo.fvControlPoints[ 0 ][ 2 ], regionControlPointInfo.fvControlPoints[ 1 ][ 2 ], fvNeighborRegionLerps[ RightSide ] );
	if( !bNeighborRegionSplits[ BottomSide ] )	regionControlPointInfo.fvControlPoints[ 2 ][ 1 ] = lerp( regionControlPointInfo.fvControlPoints[ 2 ][ 0 ], regionControlPointInfo.fvControlPoints[ 2 ][ 1 ], fvNeighborRegionLerps[ BottomSide ] );
	
	return regionControlPointInfo;

} // end RegionControlPointInfo CalculateRegionControlPointInfo( const VS_REGION_CONTROL_POINT_OUTPUT input )