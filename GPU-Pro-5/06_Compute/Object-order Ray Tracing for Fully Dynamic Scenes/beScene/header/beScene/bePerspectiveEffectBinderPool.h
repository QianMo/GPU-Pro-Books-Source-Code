/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_PERSPECTIVE_EFFECT_BINDER_POOL
#define BE_SCENE_PERSPECTIVE_EFFECT_BINDER_POOL

#include "beScene.h"
#include <beCore/beShared.h>
#include <beGraphics/Any/beAPI.h>
#include <lean/smart/com_ptr.h>
#include <vector>

namespace beScene
{

class Perspective;

/// Perspective constant buffer layout.
struct PerspectiveConstantBuffer
{
	float4 ViewProj[4][4];		///< View-projection matrix.
	float4 ViewProjInv[4][4];	///< View-projection matrix inverse.

	float4 View[4][4];			///< View matrix.
	float4 ViewInv[4][4];		///< View matrix inverse.

	float4 Proj[4][4];			///< Projection matrix.
	float4 ProjInv[4][4];		///< Projection matrix inverse.
	
	float4 CamRight[4];			///< Camera right.
	float4 CamUp[4];			///< Camera up.
	float4 CamDir[4];			///< Camera direction.
	float4 CamPos[4];			///< Camera position.
	
	float4 NearPlane;			///< Frustum near plane.
	float4 FarPlane;			///< Frusum far plane.

	float4 Time;				///< Time.
	float4 TimeStep;			///< Time Step.
};

/// Perspective effect binder pool.
class PerspectiveEffectBinderPool : public beCore::Resource
{
public:
	typedef std::vector< lean::com_ptr<beGraphics::Any::API::Buffer> > buffer_vector;

private:
	lean::com_ptr<beGraphics::Any::API::Device> m_pDevice;
	buffer_vector m_buffers;

	size_t m_currentIndex;
	beGraphics::Any::API::Buffer *m_pCurrentBuffer;
	const Perspective *m_pCurrentPerspective;

public:
	/// Constructor.
	BE_SCENE_API PerspectiveEffectBinderPool(beGraphics::Any::API::Device *pDevice, uint4 bufferCount = 4);
	/// Destructor.
	BE_SCENE_API ~PerspectiveEffectBinderPool();

	/// Gets a constant buffer filled with information on the given perspective.
	BE_SCENE_API beGraphics::Any::API::Buffer* GetPerspectiveConstants(const Perspective *pPerspective, beGraphics::Any::API::DeviceContext *pContext);

	/// Invalidates all cached perspective data.
	BE_SCENE_API void Invalidate();
};

} // namespace

#endif