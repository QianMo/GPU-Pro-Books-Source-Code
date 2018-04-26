#include "DXUT.h"
#include "TaskBlock.h"
#include "xmlParser.h"
#include "RenderCueable.h"
#include "ClearTarget.h"
#include "Play.h"
#include "Theatre.h"
#include "ClearDepthStencil.h"
#include "SetShaderResource.h"
#include "SetConstantBuffer.h"
#include "SetTargets.h"
#include "SetStreamTargets.h"
#include "SetVertexBuffers.h"
#include "TaskLoop.h"
#include "ResourceLocator.h"
#include "Cueable.h"
#include "ResourceSet.h"
#include "ScriptVariableClass.h"
#include "CopyResource.h"
#include "PropsMaster.h"
#include "ScriptResourceVariable.h"

TaskBlock::TaskBlock(void)
{
}

TaskBlock::~TaskBlock(void)
{
	taskList.deleteAll();
}

void TaskBlock::loadTasks(Play* play, XMLNode& scriptNode, ResourceOwner* localResourceOwner)
{
	if(scriptNode.isEmpty())
		return;
	ResourceLocator* resourceLocator = play->getTheatre()->getResourceLocator();
	int iTask = 0;
	XMLNode taskNode;
	while( !(taskNode = scriptNode.getChildNode(iTask)).isEmpty() )
	{
		if(wcscmp(taskNode.getName(), L"render") == 0)
		{
			const wchar_t* roleName = taskNode|L"role";
			if(roleName == NULL) EggXMLERR(taskNode, L"No role specified in Task.");
			const Role role = play->getRole(roleName);
			if(!role.isValid())
				EggXMLERR(taskNode, L"Task using undefined role type. [" << roleName << "]");
			const wchar_t* cue = taskNode|L"cue";
			const wchar_t* cameraCue = taskNode|L"cameraCue";

			if(cue)
			{
				Cueable* cueable = play->getCueable(cue);
				Cueable* cameraCueable = NULL;
				if(cameraCue)
					cameraCueable = play->getCueable(cameraCue);
				if(cueable)
					taskList.push_back( new RenderCueable(cueable, cameraCueable, role));
				else
					EggXMLERR(taskNode, L"Cue target not found.[" << cue << L"]");
			}
			else
				EggXMLERR(taskNode, L"Cue not specified.");
		}
		if(wcscmp(taskNode.getName(), L"clearRenderTargetView") == 0)
		{
			const wchar_t* rtvName = taskNode|L"renderTargetView";
			D3DXVECTOR4 color = taskNode.readVector4(L"color");
			if(rtvName)
			{
				ScriptRenderTargetViewVariable* rtvv = resourceLocator->getRenderTargetViewVariable(rtvName, localResourceOwner);
				if(rtvv)
					taskList.push_back( new ClearTarget(rtvv, color));
				else
					EggXMLERR(taskNode, L"Undefined script variable. Type: RenderTargetView. " << rtvName);
			}
			else
				EggXMLERR(taskNode, L"Render target view not specified.");
		}
		if(wcscmp(taskNode.getName(), L"clearDepthStencilView") == 0)
		{
			const wchar_t* dsvName = taskNode|L"depthStencilView";
			double depth = taskNode.readDouble(L"depth", 1);
			unsigned char stencilRef = taskNode.readLong(L"stencil", 0);
			bool clearDepth = taskNode.readBool(L"clearDepth", true);
			bool clearStencil = taskNode.readBool(L"clearStencil", false);
			if(dsvName)
			{
				ScriptDepthStencilViewVariable* dsvv = resourceLocator->getDepthStencilViewVariable(dsvName, localResourceOwner);
				if(dsvv)
				{
					taskList.push_back( new ClearDepthStencil(dsvv,
						(clearDepth?D3D10_CLEAR_DEPTH:0) | (clearStencil?D3D10_CLEAR_STENCIL:0), depth, stencilRef));
				}
				else
					EggXMLERR(taskNode, L"Undefined script variable. Type: DepthStencilView. " << dsvName);
			}
			else
				EggXMLERR(taskNode, L"Depth stencil view not specified.");
		}
		if(wcscmp(taskNode.getName(), L"setShaderResource") == 0)
		{
			std::wstring effectName = taskNode.readWString(L"effect");
			std::string effectVariableName = taskNode.readString(L"effectVariableName");
			const wchar_t* srvName = taskNode|L"shaderResourceView";
			ID3D10EffectVariable* esv = play->getTheatre()->getEffect(effectName)->GetVariableByName(effectVariableName.c_str());
			if(esv && srvName)
			{
				ScriptShaderResourceViewVariable* srv = resourceLocator->getShaderResourceViewVariable(srvName, localResourceOwner);
				if(srv)
					taskList.push_back( new SetShaderResource(
						esv->AsShaderResource(), srv ));
				else
					EggXMLERR(taskNode, L"Undefined script variable. Type: ShaderResourceView. " << srvName);
			}
			else
				EggXMLERR(taskNode, L"Shader resource view not specified.");
		}
		if(wcscmp(taskNode.getName(), L"setConstantBuffer") == 0)
		{
			std::wstring effectName = taskNode.readWString(L"effect");
			std::string effectVariableName = taskNode.readString(L"effectConstantBufferName");
			const wchar_t* resourceName = taskNode|L"buffer";
			ID3D10EffectConstantBuffer* ecb = play->getTheatre()->getEffect(effectName)->GetConstantBufferByName(effectVariableName.c_str());
			if(ecb && resourceName)
			{
				ScriptResourceVariable* resource = resourceLocator->getResourceVariable(resourceName, localResourceOwner);
				if(resource)
					taskList.push_back( new SetConstantBuffer(
						ecb, resource ));
				else
					EggXMLERR(taskNode, L"Undefined script variable. Type: Resource. " << resourceName);
			}
			else
				EggXMLERR(taskNode, L"Constant buffer not specified.");
		}
		if(wcscmp(taskNode.getName(), L"setTargets") == 0)
		{
			const wchar_t* dsvName = taskNode|L"depthStencilView";
			if(dsvName)
			{
				ScriptDepthStencilViewVariable* dsvv = resourceLocator->getDepthStencilViewVariable(dsvName, localResourceOwner);
				SetTargets* setTargets = new SetTargets(dsvv);
				int iRenderTarget = 0;
				XMLNode renderTargetNode;
				while( !(renderTargetNode = taskNode.getChildNode(iRenderTarget)).isEmpty() )
				{
					const wchar_t* rtvName = renderTargetNode|L"renderTargetView";
					ScriptRenderTargetViewVariable* rtvv = resourceLocator->getRenderTargetViewVariable(rtvName, localResourceOwner);
					setTargets->addRenderTargetView(rtvv);
					iRenderTarget++;
				}
				taskList.push_back( setTargets);
			}
			else
				EggXMLERR(taskNode, L"DepthStencilView not specified.");
		}
		if(wcscmp(taskNode.getName(), L"setStreamTargets") == 0)
		{
			SetStreamTargets* setStreamTargets = new SetStreamTargets();
			int iTarget = 0;
			XMLNode targetNode;
			while( !(targetNode = taskNode.getChildNode(iTarget)).isEmpty() )
			{
				const wchar_t* bufferName = targetNode|L"buffer";
				ScriptResourceVariable* buffer = resourceLocator->getResourceVariable(bufferName, localResourceOwner);
				unsigned int offset = targetNode.readLong(L"offset");
				setStreamTargets->addBuffer(buffer, offset);
				iTarget++;
			}
			taskList.push_back( setStreamTargets);
		}
		if(wcscmp(taskNode.getName(), L"setVertexBuffers") == 0)
		{
			SetVertexBuffers* setVertexBuffers = new SetVertexBuffers();
			int iInput = 0;
			XMLNode inputNode;
			while( !(inputNode = taskNode.getChildNode(iInput)).isEmpty() )
			{
				const wchar_t* bufferName = inputNode|L"buffer";
				ScriptResourceVariable* buffer = resourceLocator->getResourceVariable(bufferName, localResourceOwner);
				unsigned int stride = inputNode.readLong(L"stride");
				unsigned int offset = inputNode.readLong(L"offset");
				setVertexBuffers->addBuffer(buffer, stride, offset);
				iInput++;
			}
			taskList.push_back( setVertexBuffers);
		}
		if(wcscmp(taskNode.getName(), L"loop") == 0)
		{
			TaskLoop* taskLoop = new TaskLoop(play, taskNode, localResourceOwner);
			taskList.push_back( taskLoop);
		}
		if(wcscmp(taskNode.getName(), L"defineVariable") == 0)
		{
			const wchar_t* type = taskNode|L"type";
			const wchar_t* name = taskNode|L"name";
			if(type)
			{
				if(name)
				{
					const ScriptVariableClass& svc = ScriptVariableClass::fromString(type);
					if(svc == ScriptVariableClass::Unknown)
						EggXMLERR(taskNode, L"Unknown variable type.");
					localResourceOwner->getResourceSet()->createVariable(svc, name);
				}
				else
					EggXMLERR(taskNode, L"Missing variable name.");
			}
			else
				EggXMLERR(taskNode, L"Missing variable type.");
		}
		if(wcscmp(taskNode.getName(), L"linkShaderResourceViewVariableToProp") == 0)
		{
			const wchar_t* name = taskNode|L"name";
			if(name)
			{
				ID3D10Resource* texture;
				D3DX10_IMAGE_LOAD_INFO loadInfo;
				loadInfo.Width = D3DX10_DEFAULT;
				loadInfo.Height = D3DX10_DEFAULT;
				loadInfo.Depth = D3DX10_DEFAULT;
				loadInfo.FirstMipLevel = D3DX10_DEFAULT;
				loadInfo.Usage = D3D10_USAGE_STAGING;
				loadInfo.BindFlags = 0;
				loadInfo.CpuAccessFlags = D3D10_CPU_ACCESS_READ;
				loadInfo.MiscFlags = D3DX10_DEFAULT;
				loadInfo.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
				loadInfo.Filter = D3DX10_DEFAULT;
				loadInfo.MipFilter = D3DX10_DEFAULT;
				loadInfo.MipLevels = 1;

				std::wstring textureFilePath = taskNode.readWString(L"file");

				HRESULT hr = D3DX10CreateTextureFromFile( 
					play->getTheatre()->getDevice(), 
					textureFilePath.c_str(), &loadInfo, NULL, &texture, NULL );
				
				localResourceOwner->getResourceSet()->addResource(name, texture, false);
			}
			else
				EggXMLERR(taskNode, L"Missing variable name.");

		}
		if(wcscmp(taskNode.getName(), L"invoke") == 0)
		{
			const wchar_t* cue = taskNode|L"cue";
			if(cue)
			{
				Cueable* cueable = play->getCueable(cue);
				if(cueable)
				{
					Invoke* invoke = cueable->createInvocation(taskNode, localResourceOwner);
					if(invoke)
						taskList.push_back( invoke);
				}
				else
					EggXMLERR(taskNode, L"Cue target not found.[" << cue << L"]");
			}
		}
		if(wcscmp(taskNode.getName(), L"copy") == 0)
		{
			const wchar_t* srcVarName = taskNode|L"source";
			const wchar_t* destVarName = taskNode|L"destination";
			if(srcVarName && srcVarName)
			{
				ScriptResourceVariable* src = resourceLocator->getResourceVariable(srcVarName, localResourceOwner);
				ScriptResourceVariable* dest = resourceLocator->getResourceVariable(destVarName, localResourceOwner);
				if(src)
				{
					if(dest)
					{
						taskList.push_back(new CopyResource(src, dest));
					}
					else
						EggXMLERR(taskNode, L"Unknown variable. " << destVarName );
				}
				else
					EggXMLERR(taskNode, L"Unknown variable. " << srcVarName );
			}
			else
				EggXMLERR(taskNode, L"Missing source or destination parameter.");
		}

		iTask++;
	}
}

void TaskBlock::execute(const TaskContext& context)
{
	TaskList::iterator iTask = taskList.begin();
	while(iTask != taskList.end())
	{
		(*iTask)->execute(context);
		iTask++;
	}
}