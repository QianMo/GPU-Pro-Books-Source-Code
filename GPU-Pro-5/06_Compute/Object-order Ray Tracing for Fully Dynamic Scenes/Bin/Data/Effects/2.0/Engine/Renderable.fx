#ifndef BE_RENDERABLE_H
#define BE_RENDERABLE_H

/// Renderable constants.
cbuffer RenderableConstants
{
#ifdef BE_RENDERABLE_INCLUDE_WORLD
	float4x4 World : World;						///< World matrix.
	float4x4 WorldInverse : WorldInverse;		///< World matrix inverse.
#endif

#ifdef BE_RENDERABLE_INCLUDE_PROJ
	float4x4 WorldViewProj : WorldViewProj;		///< World-view-projection matrix.
#endif
#ifdef BE_RENDERABLE_INCLUDE_VIEW
	float4x4 WorldView : WorldView;				///< World-view matrix.
#endif

#ifdef BE_RENDERABLE_INCLUDE_CAM
	float3 ObjectCamPos : ObjectCamPos;			///< Object-space camera position.
	float3 ObjectCamDir : ObjectCamDir;			///< Object-space camera direction.
#endif

#ifdef BE_RENDERABLE_INCLUDE_ID
	uint ObjectID : ID;
#endif
};

#endif