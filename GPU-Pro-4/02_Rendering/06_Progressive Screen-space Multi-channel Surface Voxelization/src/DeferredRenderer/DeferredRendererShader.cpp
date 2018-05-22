#include "shaders/DeferredRendererShader.h"
#include "SceneGraph_Aux.h"
#include "DeferredRenderer.h"

stack<int> DRShader::_shader_stack = stack<int>();

void DRShader::pushShader()
{
	int prog;
	glGetIntegerv( GL_CURRENT_PROGRAM, &prog );
	_shader_stack.push(prog); 
}

void DRShader::popShader()
{
	if (!_shader_stack.empty())
	{
		glUseProgram(_shader_stack.top());
		_shader_stack.pop();
	}
	else
	{
		glUseProgram(0);
	}
}

DRShader::DRShader()
{
	shader = NULL;
}

DRShader::~DRShader()
{
	if (shader)
		shader_manager.free(shader);
}

void DRShader::start()
{
	pushShader();
	if (shader)
		shader->begin();
}

void DRShader::stop()
{
	if (shader)
		shader->end();
	popShader();
}

bool DRShader::init(DeferredRenderer* _renderer)
{
	renderer = _renderer;
	shader_manager = renderer->getShaderManager();
	if (shader)
		shader_manager.free(shader);
	return true;
}
