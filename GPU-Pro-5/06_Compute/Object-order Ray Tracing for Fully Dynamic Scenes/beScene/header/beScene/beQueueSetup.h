/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_QUEUE_SETUP
#define BE_SCENE_QUEUE_SETUP

#include "beScene.h"
#include <beCore/beShared.h>

namespace beScene
{

// Prototypes.
class RenderContext;
class Perspective;

/// Queue setup interface.
class QueueSetup : public beCore::OptionalResource
{
protected:
	QueueSetup& operator =(const QueueSetup&) { return *this; }

public:
	virtual ~QueueSetup() throw() { }

	/// Called before drawing of a specific render queue begins.
	virtual void SetupRendering(uint4 stageID, uint4 queueID, const Perspective &perspective, const RenderContext &context) const = 0;
};

} // namespace

#endif