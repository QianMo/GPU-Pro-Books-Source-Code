#include "DXUT.h"
#include "Auto.h"
#include "Theatre.h"
#include "RenderContext.h"
#include "XMLparser.h"

Auto::Auto(Theatre* theatre, XMLNode& xMainNode)
:Cueable(theatre)
{
	loadAutoRenditions(theatre, xMainNode);
}

Auto::~Auto(void)
{
	AutoRenditionDirectory::iterator i = autoRenditionDirectory.begin();
	while(i != autoRenditionDirectory.end())
	{
		i->second.inputLayout->Release();
		i++;
	}
}

void Auto::render(const RenderContext& context)
{
	AutoRenditionDirectory::iterator iAutoRendition= autoRenditionDirectory.find(context.role);
	if(iAutoRendition != autoRenditionDirectory.end())
	{
		ID3D10Device* device = getTheatre()->getDevice();

		AutoRendition& autoRendition = iAutoRendition->second;
		device->IASetInputLayout(autoRendition.inputLayout);
		device->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_POINTLIST);
		D3D10_TECHNIQUE_DESC techniqueDesc;
		autoRendition.technique->GetDesc( &techniqueDesc );
	    for(unsigned int p=0; p < techniqueDesc.Passes; p++)
	    {
			autoRendition.technique->GetPassByIndex(p)->Apply(0);

/*			D3D10_QUERY_DESC queryDesc;
			queryDesc.Query = D3D10_QUERY_PIPELINE_STATISTICS;
			queryDesc.MiscFlags = 0;
			ID3D10Query * pQuery;
			device->CreateQuery(&queryDesc, &pQuery);
			pQuery->Begin();*/

			//device->Draw(1000, 0);
			device->DrawAuto();

/*			pQuery->End();
			D3D10_QUERY_DATA_PIPELINE_STATISTICS queryData; // This data type is different depending on the query type
			while( S_OK != pQuery->GetData(&queryData, sizeof(D3D10_QUERY_DATA_PIPELINE_STATISTICS), 0) ){}
			pQuery->Release();*/
        }
	}
}

void Auto::animate(double dt, double t)
{}

void Auto::control(const ControlContext& context)
{}

void Auto::processMessage( const MessageContext& context)
{ }

Camera* Auto::getCamera()
{
	return NULL;
}

Node* Auto::getInteractors()
{
	return NULL;
}

void Auto::loadAutoRenditions(Theatre* theatre, XMLNode& xMainNode)
{
	int iAutoRendition = 0;
	XMLNode autoRenditionNode;
	while( !(autoRenditionNode = xMainNode.getChildNode(L"AutoRendition", iAutoRendition)).isEmpty() )
	{
		const wchar_t* roleName = autoRenditionNode|L"role";
		if(roleName)
		{
			Role role = theatre->getPlay()->getRole(roleName);
			if(role.isValid())
			{
				AutoRendition autoRendition;

				std::wstring effectName = autoRenditionNode.readWString(L"effect");
				std::string techniqueName = autoRenditionNode.readString(L"technique");

				autoRendition.technique = theatre->getTechnique(effectName, techniqueName);
				D3D10_PASS_DESC passDesc;
				autoRendition.technique->GetPassByIndex(0)->GetDesc(&passDesc);

				//todo
				D3D10_INPUT_ELEMENT_DESC elements[2] =
				{
					{
						"POSITION", 0,
							DXGI_FORMAT_R32G32B32A32_FLOAT,
							0, 0, D3D10_INPUT_PER_VERTEX_DATA, 0}
				,
					{
						"DIRECTION", 0,
							DXGI_FORMAT_R32G32B32A32_FLOAT,
							0, D3D10_APPEND_ALIGNED_ELEMENT, D3D10_INPUT_PER_VERTEX_DATA, 0}};

				theatre->getDevice()->CreateInputLayout(elements, 2, passDesc.pIAInputSignature, passDesc.IAInputSignatureSize, 
					&autoRendition.inputLayout);
				
				autoRenditionDirectory[role] = autoRendition;
			}
		}
		iAutoRendition++;
	}
}