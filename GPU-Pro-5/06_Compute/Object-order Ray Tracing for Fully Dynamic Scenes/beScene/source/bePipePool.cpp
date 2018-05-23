/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/bePipePool.h"

namespace beScene
{

// Constructor.
PipePool::PipePool(beGraphics::TextureTargetPool *pTargetPool)
	: m_pTargetPool( LEAN_ASSERT_NOT_NULL(pTargetPool) )
{
}

// Destructor.
PipePool::~PipePool()
{
}

// Gets a pipe.
Pipe* PipePool::GetPipe(const beGraphics::TextureTargetDesc &desc)
{
	for (pipe_vector::const_iterator it = m_pipes.begin(); it != m_pipes.end(); ++it)
	{
		Pipe *pPipe = *it;

		if (pPipe->ref_count() == 1)
		{
			pPipe->Reset(desc);
			return *it;
		}
	}

	lean::resource_ptr<Pipe> pNewPipe = CreatePipe(desc, m_pTargetPool);
	m_pipes.push_back(pNewPipe);
	return pNewPipe;
}

} // namespace
