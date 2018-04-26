#include "DXUT.h"
#include "ControlTask.h"

ControlTask::ControlTask(Cueable* cue, Cueable* interactorCue)
{
	this->cue = cue;
	this->interactorCue = interactorCue;
}

