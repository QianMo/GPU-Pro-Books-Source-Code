/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include <beGraphics/beTexture.h>

#include <beCore/beComponentReflector.h>
#include <beCore/beComponentTypes.h>

#include "beScene/beSerializationParameters.h"
#include "beScene/beResourceManager.h"

#include <beGraphics/beTextureCache.h>

#include <lean/logging/log.h>
#include <lean/logging/errors.h>

namespace beScene
{

/// Reflects textures for use in component-based editing environments.
class TextureReflector : public beCore::ComponentReflector
{
	/// Gets principal component flags.
	uint4 GetComponentFlags() const LEAN_OVERRIDE
	{
		return bec::ComponentFlags::Filed;
	}
	/// Gets specific component flags.
	uint4 GetComponentFlags(const lean::any &component) const LEAN_OVERRIDE
	{
		uint4 flags = bec::ComponentFlags::NameMutable; // | bec::ComponentFlags::FileMutable

		if (const beg::TextureView *view = any_cast_default<beg::TextureView*>(component))
			if (const beg::TextureCache *cache = view->GetCache())
				if (const beg::Texture *texture = cache->GetTexture(view))
					if (!cache->GetFile(texture).empty())
						flags |= bec::ComponentState::Filed;

		return flags;
	}

	/// Gets information on the components currently available.
	bec::ComponentInfoVector GetComponentInfo(const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		return GetSceneParameters(parameters).ResourceManager->TextureCache()->GetInfo();
	}

	/// Gets the component name.
	bec::ComponentInfo GetInfo(const lean::any &component) const LEAN_OVERRIDE
	{
		bec::ComponentInfo result;

		if (const beg::TextureView *view = any_cast_default<beg::TextureView*>(component))
			if (const beg::TextureCache *cache = view->GetCache())
				if (const beg::Texture *texture = cache->GetTexture(view))
					result = cache->GetInfo(texture);

		return result;
	}

	/// Gets a component by name.
	lean::cloneable_obj<lean::any, true> GetComponentByName(const utf8_ntri &name, const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);

		return bec::any_resource_t<beGraphics::TextureView>::t(
				sceneParameters.ResourceManager->TextureCache()->GetView(
						sceneParameters.ResourceManager->TextureCache()->GetByName(name)
					)
			);
	}

	/// Sets the component name.
	void SetName(const lean::any &component, const utf8_ntri &name) const LEAN_OVERRIDE
	{
		if (beg::TextureView *view = any_cast_default<beg::TextureView*>(component))
			if (beg::TextureCache *cache = view->GetCache())
				if (beg::Texture *texture = cache->GetTexture(view))
				{
					cache->SetName(texture, name);
					return;
				}

		LEAN_THROW_ERROR_CTX("Unknown texture cannot be renamed", name.c_str());
	}

	/// Gets a fitting file extension, if available.
	utf8_ntr GetFileExtension() const LEAN_OVERRIDE
	{
		return utf8_ntr("dds");
	}
	
	/// Gets a component from the given file.
	lean::cloneable_obj<lean::any, true> GetComponentByFile(const utf8_ntri &file,
		const beCore::Parameters &fileParameters, const beCore::ParameterSet &parameters) const LEAN_OVERRIDE
	{
		SceneParameters sceneParameters = GetSceneParameters(parameters);

		return bec::any_resource_t<beGraphics::TextureView>::t(
				sceneParameters.ResourceManager->TextureCache()->GetViewByFile(file) // TODO: Where to get parameters from? --> create?
			);
	}

	/// Gets the component type reflected.
	const beCore::ComponentType* GetType() const LEAN_OVERRIDE
	{
		return beg::TextureView::GetComponentType(); 
	}
};

static const beCore::ComponentReflectorPlugin<TextureReflector> TextureReflectorPlugin(beg::TextureView::GetComponentType());

} // namespace
