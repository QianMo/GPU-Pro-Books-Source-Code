/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beRenderContext.h"
#include <beGraphics/Any/beDeviceContext.h>
#include <beGraphics/Any/beStateManager.h>

namespace beScene
{

namespace
{

/// Clones a device context wrapper.
beGraphics::DeviceContext* CloneContext(const beGraphics::DeviceContext &context)
{
	return new beGraphics::Any::DeviceContext( ToImpl(context) );
}

// Creates a state manager.
lean::resource_ptr<beGraphics::StateManager, true> CreateStateManager(const beGraphics::DeviceContext &context)
{
	return lean::bind_resource( new beGraphics::Any::StateManager( ToImpl(context) ) );
}

}

// Constructor.
RenderContext::RenderContext(const beGraphics::DeviceContext &context, beGraphics::StateManager *pStateManager)
	: m_pContext( CloneContext(context) ),
	m_pStateManager( LEAN_ASSERT_NOT_NULL(pStateManager) )
{
}

// Copy constructor.
RenderContext::RenderContext(const RenderContext &right)
	: m_pContext( CloneContext(*right.m_pContext)  ),
	m_pStateManager( right.m_pStateManager )
{
}

// Destructor.
RenderContext::~RenderContext()
{
}

// Creates a render context from the given device context.
lean::resource_ptr<RenderContext, true> CreateRenderContext(const beGraphics::DeviceContext &context)
{
	return CreateRenderContext(context, CreateStateManager(context).get());
}

// Creates a render context from the given device context.
lean::resource_ptr<RenderContext, true> CreateRenderContext(const beGraphics::DeviceContext &context, beGraphics::StateManager *pStateManager)
{
	return lean::bind_resource( new RenderContext(context, pStateManager) );
}

} // namespace
