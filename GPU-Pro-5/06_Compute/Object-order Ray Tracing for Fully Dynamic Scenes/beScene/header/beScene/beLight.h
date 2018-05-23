/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_LIGHT
#define BE_SCENE_LIGHT

#include "beScene.h"
#include <beGraphics/beBuffer.h>
#include <beGraphics/beTexture.h>
#include <beCore/beIdentifiers.h>
#include <lean/smart/resource_ptr.h>

#include <beMath/beSphereDef.h>

namespace beScene
{

// TODO: Remove legacy interfaces

// Prototypes.
class Perspective;
class RenderContext;

/// Light flags enumeration.
struct LightFlags
{
	/// Enumeration
	enum T
	{
		PerspectivePrepare = 0x1,	///< Indicates @code Prepare(Perspective*)@endcode should be called.
		PerspectiveFinalize = 0x2,	///< Indicates @code Finalize(Perspective*)@endcode should be called.

		Shadowed = 0x4				///< Indicates shadow maps are available.
	};
	LEAN_MAKE_ENUM_STRUCT(LightFlags)
};

/// Light base.
class Light
{
protected:
	const uint4 m_lightTypeID;									///< Light type ID.
	lean::resource_ptr<const beGraphics::Buffer> m_pConstants;	///< Buffer.
	beMath::fsphere3 m_bounds;									///< Bounding sphere.
	uint4 m_flags;												///< Flags.
	
	Light& operator =(const Light&) { return *this; }
	~Light() { }

public:
	/// Constructor.
	Light(uint4 lightTypeID, const beGraphics::Buffer *pConstants = nullptr, uint4 flags = 0)
		: m_lightTypeID(lightTypeID),
		m_pConstants(pConstants),
		m_flags(flags) { }

	/// Prepares this light.
	virtual void* Prepare(Perspective &perspective) const { return nullptr; }
	/// Prepares this light.
	virtual void Finalize(Perspective &perspective, void *pData) const { }

	/// Gets shadow maps.
	virtual const beGraphics::TextureViewHandle* GetShadowMaps(const void *pData, uint4 &count) const { count = 0; return nullptr; }
	/// Gets light maps.
	virtual const beGraphics::TextureViewHandle* GetLightMaps(const void *pData, uint4 &count) const { count = 0; return nullptr; }

	/// Gets the light type ID.
	LEAN_INLINE uint4 GetLightTypeID() const { return m_lightTypeID; }
	/// Gets the constants.
	LEAN_INLINE const beGraphics::Buffer& GetConstants() const { return *m_pConstants; }
	/// Gets the bounding sphere.
	LEAN_INLINE const beMath::fsphere3& GetBounds() const { return m_bounds; } 
	/// Gets the flags.
	LEAN_INLINE uint4 GetLightFlags() const { return m_flags; }
};

/// Data-enhanced light.
struct LightJob
{
	const Light *Light;
	const void *Data;

	/// NON-INITIALIZING constructor.
	LEAN_INLINE LightJob() { }
	/// Constructor.
	LEAN_INLINE LightJob(const class Light *pLight,
		const void *pData)
			: Light(pLight),
			Data(pData) { }
};

/// Light type mapping.
typedef beCore::Identifiers LightTypes;
/// Gets the global light type mapping.
BE_SCENE_API LightTypes& GetLightTypes();

} // namespace

#endif