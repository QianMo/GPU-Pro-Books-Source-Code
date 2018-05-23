/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#include "stdafx.h"

#include "FreeCamera.h"

#include <beMath/beVector.h>
#include <beMath/beMatrix.h>
#include <beMath/beUtility.h>

#include <Windows.h>
#include <lean/logging/log.h>

namespace app
{

namespace
{
	/// Gets the cursor position.
	bem::ivec2 GetCursorPos()
	{
		POINT cursorPoint;
		::GetCursorPos(&cursorPoint);
		return bem::vec<int>(cursorPoint.x, cursorPoint.y);
	}

	/// Sets the cursor position.
	void SetCursorPos(const bem::ivec2 &pos)
	{
		::SetCursorPos(pos[0], pos[1]);
	}

} // namespace

// Constructor.
FreeCamera::FreeCamera(bees::Entity *pEntity)
	: m_pEntity( LEAN_ASSERT_NOT_NULL(pEntity) ),
	m_cursorPos( GetCursorPos() ),
	m_bMouseCaptured(false),
	m_bFree(true)
{
}

// Destructor.
FreeCamera::~FreeCamera()
{
}

// Moves the camera.
void FreeCamera::Move(float timeStep, const beLauncher::KeyboardState &input)
{
	bem::fvec3 pos = m_pEntity->GetPosition();
	const bem::fmat3 &orientation = m_pEntity->GetOrientation();

	float speed = 20.0f;
	
	if (Pressed(input, VK_CONTROL))
		speed = 2.8f;
	if (Pressed(input, VK_SHIFT))
		speed = 90.0f;
	
	const float step = speed * timeStep;

	if (Pressed(input, 'W') || Pressed(input, VK_UP))
		pos += orientation[2] * step;
	if (Pressed(input, 'S') || Pressed(input, VK_DOWN))
		pos -= orientation[2] * step;
	
	if (Pressed(input, 'A') || Pressed(input, VK_LEFT))
		pos -= orientation[0] * step;
	if (Pressed(input, 'D') || Pressed(input, VK_RIGHT))
		pos += orientation[0] * step;

	if (Pressed(input, VK_SPACE) || Pressed(input, 'E') || Pressed(input, VK_PRIOR))
		pos += orientation[1] * step;
	if (Pressed(input, 'Q') || Pressed(input, 'X') || Pressed(input, VK_NEXT))
		pos -= orientation[1] * step;

	m_pEntity->SetPosition(pos);
}

// Rotates the camera.
void FreeCamera::Rotate(float timeStep, const beLauncher::MouseState &input)
{
	float rotYSpeed = 1.0f / 2500.0f;
	float rotXSpeed = 1.0f / 1500.0f;

	bem::fvec2 cursorDelta = bem::vec(input.PosDeltaX, input.PosDeltaY);
	bem::fvec2 rotationDelta = bem::vec(
			  cursorDelta.y * sqrtf(fabs(cursorDelta.y)) * rotXSpeed
			, cursorDelta.x * sqrtf(fabs(cursorDelta.x)) * rotYSpeed
		);

	bem::fmat3 orientation = m_pEntity->GetOrientation();

	// Rotate
	orientation = mul( orientation, beMath::mat_rot_y<3>(rotationDelta.y * bem::sign(orientation[1].y + 0.5f)) );
	orientation = mul( beMath::mat_rot_x<3>(rotationDelta.x), orientation );

	// Re-orientate & orthogonalize
	orientation[1] = normalize( cross(orientation[2], beMath::vec(orientation[0].x, 0.0f, orientation[0].z)) );
	orientation[0] = normalize( cross(orientation[1], orientation[2]) );
	orientation[2] = normalize( orientation[2] );

	m_pEntity->SetOrientation(orientation);
}

// Steps the camera.
void FreeCamera::Step(float timeStep, const beLauncher::InputState &input)
{
	if (JustPressed(*input.KeyState, VK_RETURN))
	{
		m_bMouseCaptured = !m_bMouseCaptured;
		m_cursorPos = GetCursorPos();
	}
	if (JustPressed(*input.KeyState, VK_ESCAPE))
		m_bMouseCaptured = false;

	if (m_bMouseCaptured || Pressed(*input.MouseState, beLauncher::MouseButton::Right))
		Rotate(timeStep, *input.MouseState);

	if (m_bMouseCaptured)
		SetCursorPos(m_cursorPos);

	if (m_bFree)
		Move(timeStep, *input.KeyState);

	if (Pressed(*input.KeyState, 'P'))
		lean::debug_stream() << "Cam pos: " << m_pEntity->GetPosition()[0] << "; " << m_pEntity->GetPosition()[1] << "; " << m_pEntity->GetPosition()[2];
}

} // namespace