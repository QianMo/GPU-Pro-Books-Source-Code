#include "DXUT.h"
#include "Nothing.h"
#include "Theatre.h"
#include "RenderContext.h"
#include "XMLparser.h"

Nothing::Nothing(Theatre* theatre, XMLNode& xMainNode)
:Cueable(theatre)
{
	loadNothingRenditions(theatre, xMainNode);
}

Nothing::~Nothing(void)
{
	NothingRenditionDirectory::iterator i = nothingRenditionDirectory.begin();
	while(i != nothingRenditionDirectory.end())
	{
		i->second.inputLayout->Release();
		i++;
	}
}

void Nothing::render(const RenderContext& context)
{
	NothingRenditionDirectory::iterator iNothingRendition = nothingRenditionDirectory.find(context.role);
	if(iNothingRendition != nothingRenditionDirectory.end())
	{
		ID3D10Device* device = getTheatre()->getDevice();

		NothingRendition& role = iNothingRendition->second;
		ID3D10Buffer* nullBuffers[2] = {NULL, NULL};
		unsigned int nullStrides[2] = {0, 0};
		device->IASetVertexBuffers(0, 2, nullBuffers, nullStrides, nullStrides);
		device->IASetInputLayout(role.inputLayout);
		device->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_POINTLIST);
		D3D10_TECHNIQUE_DESC techniqueDesc;
		role.technique->GetDesc( &techniqueDesc );
	    for(unsigned int p=0; p < techniqueDesc.Passes; p++)
	    {
			role.technique->GetPassByIndex(p)->Apply(0);

			//D3D10_QUERY_DESC queryDesc;
			//queryDesc.Query = D3D10_QUERY_SO_STATISTICS;
			//queryDesc.MiscFlags = 0;
			//ID3D10Query * pQuery;
			//device->CreateQuery(&queryDesc, &pQuery);
			//pQuery->Begin();

			device->Draw(role.vertexCount, 0);

			//pQuery->End();
			//D3D10_QUERY_DATA_SO_STATISTICS queryData; // This data type is different depending on the query type
			//while( S_OK != pQuery->GetData(&queryData, sizeof(D3D10_QUERY_DATA_SO_STATISTICS), 0) ){}
			//pQuery->Release();
			//bool kamu = true;
        }
	}
}

void Nothing::animate(double dt, double t)
{}

void Nothing::control(const ControlContext& context)
{}

void Nothing::processMessage( const MessageContext& context)
{}

Camera* Nothing::getCamera()
{
	return NULL;
}

Node* Nothing::getInteractors()
{
	return NULL;
}

void Nothing::loadNothingRenditions(Theatre* theatre, XMLNode& xMainNode)
{
	int iNothingRendition = 0;
	XMLNode nothingRenditionNode;
	while( !(nothingRenditionNode = xMainNode.getChildNode(L"NothingRendition", iNothingRendition)).isEmpty() )
	{
		const wchar_t* roleName = nothingRenditionNode|L"role";
		if(roleName)
		{
			Role role = theatre->getPlay()->getRole(roleName);
			if(role.isValid())
			{
				NothingRendition nothingRendition;

				nothingRendition.vertexCount = nothingRenditionNode.readLong(L"vertexCount");

				std::wstring effectName = nothingRenditionNode.readWString(L"effect");
				std::string techniqueName = nothingRenditionNode.readString(L"technique");

				nothingRendition.technique = theatre->getTechnique(effectName, techniqueName);

				D3D10_PASS_DESC passDesc;
				nothingRendition.technique->GetPassByIndex(0)->GetDesc(&passDesc);

				D3D10_INPUT_ELEMENT_DESC elements[1];
				theatre->getDevice()->CreateInputLayout(elements, 0, passDesc.pIAInputSignature, passDesc.IAInputSignatureSize, 
					&nothingRendition.inputLayout);
				
				nothingRenditionDirectory[role] = nothingRendition;
			}
		}
		iNothingRendition++;
	}
}