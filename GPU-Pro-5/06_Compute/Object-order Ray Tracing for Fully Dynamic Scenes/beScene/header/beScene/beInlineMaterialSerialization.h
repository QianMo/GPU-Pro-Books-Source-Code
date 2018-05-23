/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_INLINE_MATERIAL_SERIALIZATION
#define BE_SCENE_INLINE_MATERIAL_SERIALIZATION

#include "beScene.h"
#include <beGraphics/beMaterial.h>
#include <beCore/beParameterSet.h>
#include <beCore/beSerializationJobs.h>

namespace beScene
{

/// Schedules the given material for inline serialization.
BE_SCENE_API void SaveMaterial(const beGraphics::Material *material,
	beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue);
/// Schedules the given material for inline serialization.
BE_SCENE_API void SaveMaterialConfig(const beGraphics::MaterialConfig *config,
	beCore::ParameterSet &parameters, beCore::SerializationQueue<beCore::SaveJob> &queue);

} // namespace

#endif