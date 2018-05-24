#include "InputManager.h"
#include "SharedContext.h"
#include "Log.h"

using namespace Log;

InputManager::InputManager()
	: m_controller(nullptr)
{
	memset(&m_triggers, 0, sizeof(m_triggers));
	memset(&m_thumbSticks, 0, sizeof(m_thumbSticks));

	m_keyState.resize(SDL_NUM_SCANCODES, KeyState::UP);
	m_prevKeyState.resize(SDL_NUM_SCANCODES, KeyState::UP);

	//Open the first available controller
	SDL_JoystickEventState(SDL_ENABLE);

	int num_joysticks = SDL_NumJoysticks();
	for (int i = 0; i < num_joysticks; ++i) 
	{
		if (SDL_IsGameController(i)) 
		{
			m_controller = SDL_GameControllerOpen(i);
			if (m_controller) 
			{
				break;
			}
			else 
			{
				PRINT(LogLevel::NON_FATAL_ERROR, "Could not open gamecontroller %d : %s", i, SDL_GetError());
			}
		}
	}
}

InputManager::~InputManager()
{
	if(m_controller)
		SDL_GameControllerClose(m_controller);
}

void InputManager::Update(SDL_Event& event)
{
	switch (event.type)
	{
	case SDL_MOUSEWHEEL:
		m_mouseScroll += event.wheel.y;
		break;
	case SDL_KEYDOWN:
		if (!event.key.repeat)
		{
			m_keyState[event.key.keysym.scancode] = KeyState::DOWN;
		}
		break;
	case SDL_KEYUP:
		m_keyState[event.key.keysym.scancode] = KeyState::UP;
		break;
	case SDL_MOUSEBUTTONDOWN:
		m_keyState[event.button.button - SDL_BUTTON_LEFT + (uint32)MouseButton::LEFT] = KeyState::DOWN;
		break;
	case SDL_MOUSEBUTTONUP:
		m_keyState[event.button.button - SDL_BUTTON_LEFT + (uint32)MouseButton::LEFT] = KeyState::UP;
		break;
	case SDL_CONTROLLERBUTTONDOWN:
		m_keyState[event.cbutton.button + (uint32)GamepadButton::A] = KeyState::DOWN;
		break;
	case SDL_CONTROLLERBUTTONUP:
		m_keyState[event.cbutton.button + (uint32)GamepadButton::A] = KeyState::UP;
		break;
	case SDL_MOUSEMOTION:
			m_globalMousePos.x = (float)event.motion.x;
			m_globalMousePos.y = (float)event.motion.y;

			m_deltaMousePos.x = (float)event.motion.xrel;
			m_deltaMousePos.y = (float)event.motion.yrel;
		break;
	case SDL_CONTROLLERAXISMOTION:
		switch (event.caxis.axis)
		{
		case SDL_CONTROLLER_AXIS_LEFTX:
			m_thumbSticks.leftX = (GAMEPAD_DEADZONE - abs(event.caxis.value) > 0) ? 0 : event.caxis.value/32768.0f;
			break;
		case SDL_CONTROLLER_AXIS_LEFTY:
			m_thumbSticks.leftY = (GAMEPAD_DEADZONE - abs(event.caxis.value) > 0) ? 0 : event.caxis.value/32768.0f;
			break;
		case SDL_CONTROLLER_AXIS_RIGHTX:
			m_thumbSticks.rightX = (GAMEPAD_DEADZONE - abs(event.caxis.value) > 0) ? 0 : event.caxis.value/32768.0f;
			break;
		case SDL_CONTROLLER_AXIS_RIGHTY:
			m_thumbSticks.rightY = (GAMEPAD_DEADZONE - abs(event.caxis.value) > 0) ? 0 : event.caxis.value/32768.0f;
			break;
		case SDL_CONTROLLER_AXIS_TRIGGERLEFT:
			m_triggers.left = event.caxis.value/32768.0f;
			break;
		case SDL_CONTROLLER_AXIS_TRIGGERRIGHT:
			m_triggers.right = event.caxis.value/32768.0f;
			break;

		default:
			break;
		}
		break;
	default:
		break;
	}
}

void InputManager::Reset()
{
	m_deltaMousePos = Vector2(0, 0);
	m_mouseScroll = 0;

	std::copy(m_keyState.begin(), m_keyState.end(), m_prevKeyState.begin());
}

KeyState InputManager::GetKeyState(SDL_Scancode key)
{
	if (m_keyState[key] == KeyState::DOWN)
	{
		if (m_prevKeyState[key] == KeyState::DOWN)
		{
			return KeyState::DOWN;
		}
		else if (m_prevKeyState[key] == KeyState::UP)
		{
			return KeyState::DOWN_EDGE;
		}
	}
	else if (m_keyState[key] == KeyState::UP)
	{
		if (m_prevKeyState[key] == KeyState::DOWN)
		{
			return KeyState::UP_EDGE;
		}
		else if (m_prevKeyState[key] == KeyState::UP)
		{
			return KeyState::UP;
		}
	}

	assert(false);
	return KeyState::UP;
}


