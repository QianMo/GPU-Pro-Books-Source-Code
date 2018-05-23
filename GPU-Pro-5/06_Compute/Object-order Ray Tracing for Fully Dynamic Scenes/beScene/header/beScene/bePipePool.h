/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_PIPE_POOL
#define BE_SCENE_PIPE_POOL

#include "beScene.h"
#include <beCore/beShared.h>
#include <beGraphics/beTextureTargetPool.h>
#include "bePipe.h"
#include <lean/smart/resource_ptr.h>
#include <vector>

namespace beScene
{

// Prototypes
class Pipe;

/// Pool of pipes.
class PipePool : public beCore::Resource
{
public:
	typedef std::vector<lean::resource_ptr<Pipe>> pipe_vector;

private:
	lean::resource_ptr<beGraphics::TextureTargetPool> m_pTargetPool;

	pipe_vector m_pipes;

public:
	/// Constructor.
	BE_SCENE_API PipePool(beGraphics::TextureTargetPool *pPool);
	/// Destructor.
	BE_SCENE_API ~PipePool();

	/// Gets a pipe.
	BE_SCENE_API Pipe* GetPipe(const beGraphics::TextureTargetDesc &desc);
};

} // namespace

#endif