/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/bePerspectivePool.h"
#include "beScene/bePipelinePerspective.h"
#include "beScene/bePipePool.h"

namespace beScene
{

// Constructor.
PerspectivePool::PerspectivePool(PipePool *pPipePool)
	: m_pPipePool(pPipePool)
{
}

// Destructor.
PerspectivePool::~PerspectivePool()
{
}

// Adds a perspective.
PipelinePerspective* PerspectivePool::GetPerspective(Pipe *pPipe, PipelineProcessor *pProcessor, PipelineStageMask stageMask)
{
	PipelinePerspective *perspective = m_pool.FreeElement();

	if (perspective)
		perspective->Reset(pPipe, pProcessor, stageMask);
	else
		perspective = m_pool.AddElement( new PipelinePerspective(pPipe, pProcessor, stageMask) );

	return perspective;
}

// Gets a perspective.
PipelinePerspective* PerspectivePool::GetPerspective(const PerspectiveDesc &desc, Pipe *pPipe,
													 PipelineProcessor *pProcessor, PipelineStageMask stageMask)
{
	PipelinePerspective *perspective = GetPerspective(pPipe, pProcessor, stageMask);
	perspective->SetDesc(desc);
	return perspective;
}

} // namespace
