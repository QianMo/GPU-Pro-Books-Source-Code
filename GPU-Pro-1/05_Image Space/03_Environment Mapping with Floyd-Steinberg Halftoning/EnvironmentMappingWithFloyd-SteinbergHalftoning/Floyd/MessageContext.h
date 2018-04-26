#pragma once

class ControlStatus;
class Camera;
#include "TaskContext.h"

class MessageContext : public TaskContext
{
public:
	const ControlStatus& controlStatus;
	Camera* camera;
	HWND hWnd;
	UINT uMsg;
	WPARAM wParam;
	LPARAM lParam;
	bool* trapped;

	MessageContext(Theatre* theatre, ResourceOwner* localResourceOwner,
		const ControlStatus& controlStatus,
		Camera* camera,
		HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* trapped)
		:TaskContext(theatre, localResourceOwner), controlStatus(controlStatus)
	{
		this->camera = camera;
		this->hWnd = hWnd;
		this->uMsg = uMsg;
		this->wParam = wParam;
		this->lParam = lParam;
		this->trapped = trapped;
	}

	const MessageContext* asMessageContext() const {return this;}
};
