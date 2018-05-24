#pragma once
class KGraphicsDevice;

struct SharedContext 
{
	KGraphicsDevice* gfx_device;
};

extern SharedContext shared_context;