#ifndef _DR_SHADER_
#define _DR_SHADER_

#include "glsl.h"
#include <stack>

using namespace std;
using namespace cwc;

class DRShader
{
protected:
	static stack<int> _shader_stack;
	class DeferredRenderer* renderer;
	glShader *shader;
	glShaderManager shader_manager;

	void pushShader();
	void popShader();

public:
	DRShader();
	virtual ~DRShader();
	virtual void start();
	virtual void stop();
	virtual bool init(class DeferredRenderer* _renderer);
};

#endif
