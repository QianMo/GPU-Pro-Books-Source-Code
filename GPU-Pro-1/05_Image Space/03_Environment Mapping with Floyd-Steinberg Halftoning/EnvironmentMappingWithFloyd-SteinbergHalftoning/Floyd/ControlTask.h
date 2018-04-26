#pragma once

class Cueable;

class ControlTask
{
	friend class Act;
	Cueable* cue;
	Cueable* interactorCue;
public:
	ControlTask(Cueable* cue, Cueable* interactorCue);
};
