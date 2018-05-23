/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beLight.h"

namespace beScene
{

// Gets the global light types mapping.
LightTypes& GetLightTypes()
{
	static beCore::Identifiers lightTypes;
	return lightTypes;
}

} // namespace