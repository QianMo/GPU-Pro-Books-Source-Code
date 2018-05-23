/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_PERSPECTIVE
#define BE_SCENE_PERSPECTIVE

#include "beScene.h"
#include <beCore/beShared.h>
#include <beCore/bePooled.h>
#include <beMath/beVectorDef.h>
#include <beMath/beMatrixDef.h>
#include <beMath/bePlaneDef.h>
#include <memory>
#include <lean/memory/chunk_heap.h>
#include <lean/smart/com_ptr.h>
#include <lean/containers/simple_vector.h>
#include <lean/tags/noncopyable.h>

namespace beScene
{
	
// Prototypes
class Pipe;

/// Perspective flags.
struct PerspectiveFlags
{
	// Enum.
	enum T
	{
		None = 0,

		Omnidirectional		/// Treats the perspective as omnidirectional, don't forget to provide custum frustum planes!
	};
	LEAN_MAKE_ENUM_STRUCT(PerspectiveFlags)
};

/// Perspective description.
struct PerspectiveDesc
{
	beMath::fvec3 CamPos;		///< Camera position.
	beMath::fvec3 CamRight;		///< Camera orientation.
	beMath::fvec3 CamUp;		///< Camera orientation.
	beMath::fvec3 CamLook;		///< Camera orientation.
	beMath::fmat4 ViewMat;		///< View matrix.
	beMath::fmat4 ProjMat;		///< Projection matrix.
	beMath::fmat4 ViewProjMat;	///< View-projection matrix.
	beMath::fplane3 Frustum[6];	///< Frustum planes.
	float NearPlane;			///< Frustum near plane distance.
	float FarPlane;				///< Frustum far plane distance.
	bool Flipped;				///< Flipped flag.
	float Time;					///< Time.
	float TimeStep;				///< Time Step.
	uint4 OutputIndex;			///< Output index.
	uint4 Flags;				///< Perspective flags.

	/// Default constructor.
	LEAN_INLINE PerspectiveDesc() { }
	/// Constructor.
	LEAN_INLINE PerspectiveDesc(
		const beMath::fvec3 &camPos,
		const beMath::fvec3 &camRight,
		const beMath::fvec3 &camUp,
		const beMath::fvec3 &camLook,
		const beMath::fmat4 &viewMat,
		const beMath::fmat4 &projMat,
		const beMath::fmat4 &viewProjMat,
		const beMath::fplane3 *frustum,
		float nearPlane,
		float farPlane,
		bool flipped,
		float time,
		float timeStep,
		uint4 out = 0,
		uint4 flags = PerspectiveFlags::None)
			: CamPos(camPos),
			CamRight(camRight),
			CamUp(camUp),
			CamLook(camLook),
			ViewMat(viewMat),
			ProjMat(projMat),
			ViewProjMat(viewProjMat),
			NearPlane(nearPlane),
			FarPlane(farPlane),
			Flipped(flipped),
			Time(time),
			TimeStep(timeStep),
			OutputIndex(out),
			Flags(flags)
	{
		if (frustum)
			memcpy(Frustum, frustum, sizeof(Frustum));
	}
	/// Constructor.
	LEAN_INLINE PerspectiveDesc(
		const beMath::fvec3 &camPos,
		const beMath::fvec3 &camRight,
		const beMath::fvec3 &camUp,
		const beMath::fvec3 &camLook,
		const beMath::fmat4 &viewMat,
		const beMath::fmat4 &projMat,
		float nearPlane,
		float farPlane,
		bool flipped,
		float time,
		float timeStep,
		uint4 out = 0,
		uint4 flags = PerspectiveFlags::None)
			: CamPos(camPos),
			CamRight(camRight),
			CamUp(camUp),
			CamLook(camLook),
			ViewMat(viewMat),
			ProjMat(projMat),
			ViewProjMat(Multiply(viewMat, projMat)),
			NearPlane(nearPlane),
			FarPlane(farPlane),
			Flipped(flipped),
			Time(time),
			TimeStep(timeStep),
			OutputIndex(out),
			Flags(flags)
	{
		ExtractFrustum(Frustum, ViewProjMat);
	}

	/// Multiplies the given two matrices.
	static BE_SCENE_API beMath::fmat4 Multiply(const beMath::fmat4 &a, const beMath::fmat4 &b);
	/// Extracts a view frustum from the given view projection matrix.
	static BE_SCENE_API void ExtractFrustum(beMath::fplane3 frustum[6], const beMath::fmat4 &viewProj);
};

/// Rendering perspective.
class Perspective : public lean::noncopyable_chain<beCore::Shared>, public beCore::PooledToRefCounted<Perspective>
{
	friend beCore::Pooled<Perspective>;

public:
	struct State
	{
		const void *Owner;
		// TODO: Enhance by specific interface that allows for expiring?
		lean::com_ptr<beCore::RefCounted> Data;

		State(const void *owner, beCore::RefCounted *data)
			: Owner(owner),
			Data(data) { }
	};

private:
	typedef lean::simple_vector<State, lean::containers::vector_policies::semipod> state_t;
	state_t m_state;

	typedef lean::chunk_heap<0, lean::default_heap, 0, 16> data_heap;
	data_heap m_dataHeap;
	size_t m_dataHeapWatermark;

	// COMPATIBILITY: Hidden to avoid confusion
	using PooledToRefCounted::Release;
	template <class COMType>
	friend void lean::smart::release_com(COMType*);

protected:
	PerspectiveDesc m_desc;		///< Perspective description.

	/// Called when all users have released their references.
	LEAN_INLINE void UsersReleased() { Reset(); }

	/// Frees all data allocated from this perspective's internal heap.
	BE_SCENE_API void FreeData();
	/// Resets this perspective after all temporary data has been discarded.
	BE_SCENE_API virtual void ResetReleased();

public:
	/// Constructor.
	BE_SCENE_API Perspective();
	/// Constructor.
	BE_SCENE_API Perspective(const PerspectiveDesc &desc);
	/// Destructor.
	BE_SCENE_API virtual ~Perspective();

	/// Resets this perspective.
	LEAN_INLINE void Reset() { ReleaseIntermediate(); ResetReleased(); }
	/// Discards temporary data.
	BE_SCENE_API virtual void ReleaseIntermediate();

	/// Updates the description of this perspective.
	BE_SCENE_API void SetDesc(const PerspectiveDesc &desc);
	/// Gets the perspective description.
	LEAN_INLINE const PerspectiveDesc& GetDesc() const { return m_desc; };

	/// Stores the given state object for the given owner.
	BE_SCENE_API void SetState(const void *owner, beCore::RefCounted *state);
	/// Retrieves the state object stored for the given owner.
	BE_SCENE_API beCore::RefCounted* GetState(const void *owner) const;

	/// Allocates data from this perspective's internal heap.
	BE_SCENE_API void* AllocateData(size_t size);

	/// Optionally gets a pipe.
	virtual Pipe* GetPipe() const { return nullptr; }
};

/// Allocates data from the given perspective's internal heap.
template <class Type>
LEAN_INLINE Type* AllocateData(Perspective &perspective)
{
	return static_cast<Type*>(perspective.AllocateData(sizeof(Type)));
}

} // namespace

/// Allocates data from the given perspective's internal heap.
LEAN_INLINE void* operator new(size_t size, beScene::Perspective &perspective) { return perspective.AllocateData(size); }
LEAN_INLINE void operator delete(void*, beScene::Perspective&) throw() { }

/// Allocates data from the given perspective's internal heap.
LEAN_INLINE void* operator new[](size_t size, beScene::Perspective &perspective) { return perspective.AllocateData(size); }
LEAN_INLINE void operator delete[](void*, beScene::Perspective&) throw() { }

#endif