#pragma once

class Node;
class ControlStatus;

/// Structure used both as an input state manager and passed as parameter to control calls.
class ControlContext
{
	void operator=(const ControlContext& o){}
public:
	const ControlStatus& controlStatus;

	/// Time step.
	double dt;
	/// Scene graph reference for interaction computations.
	Node* interactors;

	ControlContext(const ControlStatus& controlStatus,
		double dt,
		Node* interactors)
		:controlStatus(controlStatus)
	{
		this->dt = dt;
		this->interactors = interactors;
	}
};
