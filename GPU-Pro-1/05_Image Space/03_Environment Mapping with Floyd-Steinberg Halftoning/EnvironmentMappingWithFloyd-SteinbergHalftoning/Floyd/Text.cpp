#include "DXUT.h"
#include "Text.h"
#include "Theatre.h"
#include "RenderContext.h"
#include "XMLparser.h"

Text::Text(Theatre* theatre, XMLNode& xMainNode)
:Cueable(theatre)
{	

	D3DX10CreateFont( theatre->getDevice(), 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET, 
		OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, 
		L"Arial", &textFont );

	D3DX10CreateSprite( theatre->getDevice(), 8, &textSprite );

	textHelper = new CDXUTTextHelper(textFont, textSprite, 15);
}

Text::~Text(void)
{
	delete textHelper;
	textFont->Release();
	textSprite->Release();
}

void Text::render(const RenderContext& context)
{
	ID3D10BlendState* blendState;
	float blendFactor[8];
	unsigned int sampleMask;
	getTheatre()->getDevice()->OMGetBlendState(&blendState, blendFactor, &sampleMask);
	ID3D10DepthStencilState* depthStencilState;
	unsigned int stencilRef;
	getTheatre()->getDevice()->OMGetDepthStencilState(&depthStencilState, &stencilRef);
	getTheatre()->getDevice()->OMSetBlendState(NULL, blendFactor, sampleMask);
	getTheatre()->getDevice()->OMSetDepthStencilState(NULL, stencilRef);
	textHelper->Begin();
	textHelper->SetInsertionPos( 2, 0 );
	textHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 0.6f, 0.4f, 1.0f ) );
	textHelper->DrawTextLine(	DXUTGetFrameStats(true) );
	textHelper->DrawTextLine( DXUTGetDeviceStats() );
	textHelper->DrawTextLine( L"Use WASD+mouse to navigate." );
	textHelper->DrawTextLine( L"Press space to change rendering mode." );
	textHelper->DrawTextLine( L"Current technique:" );
	textHelper->DrawTextLine( context.theatre->getPlay()->getCurrentActName().c_str());
	textHelper->End();
	getTheatre()->getDevice()->OMSetBlendState(blendState, blendFactor, sampleMask);
	getTheatre()->getDevice()->OMSetDepthStencilState(depthStencilState, stencilRef);
	if(blendState)
		blendState->Release();
	if(depthStencilState)
		depthStencilState->Release();
}

void Text::animate(double dt, double t)
{
}

void Text::control(const ControlContext& context)
{
}

void Text::processMessage( const MessageContext& context)
{
}

Camera* Text::getCamera()
{
	return NULL;
}

Node* Text::getInteractors()
{
	return NULL;
}