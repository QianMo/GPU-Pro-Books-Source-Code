#pragma once

#include "Tracing.h"

#include <beLauncher/beInput.h>
#include <beEntitySystem/beEntities.h>
#include <lean/smart/resource_ptr.h>

namespace app
{

/// Sample 2.
class FreeCamera
{
private:
	bees::Entity* m_pEntity;

	bem::ivec2 m_cursorPos;

	bool m_bMouseCaptured;
	bool m_bFree;
	
	/// Moves the camera.
	void Move(float timeStep, const beLauncher::KeyboardState &input);
	/// Rotates the camera.
	void Rotate(float timeStep, const beLauncher::MouseState &input);

public:
	/// Constructor.
	FreeCamera(bees::Entity *pEntity);
	/// Destructor.
	~FreeCamera();

	/// Steps the camera.
	void Step(float timeStep, const beLauncher::InputState &input);

	/// Enables free navigation.
	void SetFree(bool bFree) { m_bFree = bFree; }
	/// Gets whether navigation is currently free.
	bool IsFree() const { return m_bFree; };

	/// Updates the window rectangle.
	void UpdateScreen(const bem::ivec2 &pos, const bem::ivec2 &ext);

	/// Gets the entity.
	bees::Entity* GetEntity() const { return m_pEntity; }
};

} // namespace