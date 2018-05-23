/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_PROCESSOR
#define BE_SCENE_PROCESSOR

#include "beScene.h"
#include <beCore/beShared.h>

namespace beScene
{

// Prototypes.
class Perspective;
class RenderContext;

/// Processor base.
class Processor : public beCore::Resource
{
protected:
	Processor& operator =(const Processor&) { return *this; }

public:
	virtual ~Processor() { }

	/// Applies this processor.
	virtual void Render(const Perspective *pPerspective, const RenderContext &context) const = 0;
};

} // namespace

#endif