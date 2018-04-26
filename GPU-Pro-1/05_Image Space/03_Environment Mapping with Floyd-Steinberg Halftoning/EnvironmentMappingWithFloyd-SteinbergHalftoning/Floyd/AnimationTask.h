#pragma once

class Cueable;

class AnimationTask
{
	friend class Act;
	Cueable* cue;
public:
	AnimationTask(Cueable* cue);
};
