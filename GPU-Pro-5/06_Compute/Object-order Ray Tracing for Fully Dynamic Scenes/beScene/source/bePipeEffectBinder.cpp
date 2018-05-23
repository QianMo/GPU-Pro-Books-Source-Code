/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/bePipeEffectBinder.h"
#include "beScene/DX11/bePipe.h"
#include <beGraphics/Any/beDeviceContext.h>
#include <beGraphics/Any/beTextureTargetPool.h>
#include <beGraphics/DX/beError.h>
#include <beMath/beVector.h>
#include <lean/io/numeric.h>
#include <lean/strings/utility.h>

namespace beScene
{

/// Destination.
struct Destination
{
	uint4 targetID;

	bool bKeep;
};

/// Destination.
struct ColorDestination : public Destination
{
	bool bClearColor;
	bool bClearColorOnce;
	float clearColor[4];
};

/// Depth-stencil Destination.
struct DepthStencilDestination : public Destination
{
	uint4 clearFlags;
	uint4 clearOnceFlags;
	float clearDepth;
	uint4 clearStencil;
};

static const size_t MaxSimultaneousDestinationCount = 4;

/// Pass.
struct PipeEffectBinder::Pass
{
	uint4 passID;
	ID3DX11EffectPass *pPass;

	ColorDestination color[MaxSimultaneousDestinationCount];
	uint4 colorCount;

	DepthStencilDestination depthStencil;

	float scaleX;
	float scaleY;

	uint4 resolutionX;
	uint4 resolutionY;

	bool bMultisampled;
	bool bMultisamplingOnly;
	bool bRevertTargets;

	/// NON-INITIALIZING Constructor.
	Pass(uint4 passID,
		ID3DX11EffectPass *pPass)
			: passID(passID),
			pPass(pPass) { }
};

/// Target type enumeration.
struct TargetType
{
	/// Enumeration
	enum T
	{
		Unknown,	///< Unknown target type.

		Temporary,	///< Temporary target.
		Permanent,	///< (Frame-)Permanent target.
		Persistent	///< Persistent target.
	};
	LEAN_MAKE_ENUM_STRUCT(TargetType)
};

/// Target.
struct PipeEffectBinder::Target
{
	ID3DX11EffectShaderResourceVariable *pTexture;
	ID3DX11EffectVectorVariable *pTextureScaling;
	ID3DX11EffectVectorVariable *pTextureResolution;
	ID3DX11EffectScalarVariable *pTextureMultisampling;

	utf8_string name;
	TargetType::T type;
	bool bPerObject;
	bool bOutput;

	bool bMultisampled;
	DXGI_FORMAT format;
	uint4 MipLevels;
	
	uint4 sourceTargetID;
	uint4 disposePassID;
};

namespace
{

/// Gets a target type from the given string.
TargetType::T GetTargetType(const utf8_ntri &type)
{
	if (type == "Temporary")
		return TargetType::Temporary;
	else if (type == "Permanent")
		return TargetType::Permanent;
	else if (type == "Persistent")
		return TargetType::Persistent;
	else 
		return TargetType::Unknown;
}

/// Gets a format from the given string.
DXGI_FORMAT GetTargetFormat(const utf8_ntri &format)
{
	if (format == "R8U")
		return DXGI_FORMAT_R8_UNORM;
	else if (format == "R8G8U")
		return DXGI_FORMAT_R8G8_UNORM;
	else if (format == "R8G8B8U")
		return DXGI_FORMAT_R8G8B8A8_UNORM;
	else if (format == "R8G8B8A8U")
		return DXGI_FORMAT_R8G8B8A8_UNORM;
	else if (format == "R8G8B8A8U_SRGB")
		return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;

	else if (format == "R16F")
		return DXGI_FORMAT_R16_FLOAT;
	else if (format == "R16G16F")
		return DXGI_FORMAT_R16G16_FLOAT;
	else if (format == "R16G16B16F")
		return DXGI_FORMAT_R16G16B16A16_FLOAT;
	else if (format == "R16G16B16A16F")
		return DXGI_FORMAT_R16G16B16A16_FLOAT;

	else if (format == "R32F")
		return DXGI_FORMAT_R32_FLOAT;
	else if (format == "R32G32F")
		return DXGI_FORMAT_R32G32_FLOAT;
	else if (format == "R32G32B32F")
		return DXGI_FORMAT_R32G32B32_FLOAT;
	else if (format == "R32G32B32A32F")
		return DXGI_FORMAT_R32G32B32A32_FLOAT;

	else if (format == "D16")
		return DXGI_FORMAT_D16_UNORM;
	else if (format == "D24S8")
		return DXGI_FORMAT_D24_UNORM_S8_UINT;
	else if (format == "D32")
		return DXGI_FORMAT_D32_FLOAT;

	else if (format == "R16UN")
		return DXGI_FORMAT_R16_UNORM;
	else if (format == "R16G16UN")
		return DXGI_FORMAT_R16G16_UNORM;
	else if (format == "R16G16B16UN")
		return DXGI_FORMAT_R16G16B16A16_UNORM;
	else if (format == "R16G16B16A16UN")
		return DXGI_FORMAT_R16G16B16A16_UNORM;

	else if (format == "R16U")
		return DXGI_FORMAT_R16_UINT;
	else if (format == "R16G16U")
		return DXGI_FORMAT_R16G16_UINT;
	else if (format == "R16G16B16U")
		return DXGI_FORMAT_R16G16B16A16_UINT;
	else if (format == "R16G16B16A16U")
		return DXGI_FORMAT_R16G16B16A16_UINT;

	else if (format == "R32U")
		return DXGI_FORMAT_R32_UINT;
	else if (format == "R32G32U")
		return DXGI_FORMAT_R32G32_UINT;
	else if (format == "R32G32B32U")
		return DXGI_FORMAT_R32G32B32_UINT;
	else if (format == "R32G32B32A32U")
		return DXGI_FORMAT_R32G32B32A32_UINT;

	else
		return DXGI_FORMAT_UNKNOWN;

}

/// Gets a scalar variable of the given name or nullptr, if unavailable.
ID3DX11EffectScalarVariable* MaybeGetScalarVariable(ID3DX11Effect *pEffect, const lean::utf8_ntri &name)
{
	ID3DX11EffectScalarVariable *pVariable = pEffect->GetVariableBySemantic(name.c_str())->AsScalar();
	return (pVariable->IsValid()) ? pVariable : nullptr;
}

/// Gets a vector variable of the given name or nullptr, if unavailable.
ID3DX11EffectVectorVariable* MaybeGetVectorVariable(ID3DX11Effect *pEffect, const lean::utf8_ntri &name)
{
	ID3DX11EffectVectorVariable *pVariable = pEffect->GetVariableBySemantic(name.c_str())->AsVector();
	return (pVariable->IsValid()) ? pVariable : nullptr;
}

/// Gets the index of the target identified by the given name.
template <class TargetIterator>
uint4 GetTargetIndex(const TargetIterator targets, const TargetIterator targetsEnd, const utf8_ntri &name)
{
	for (TargetIterator target = targets; target != targetsEnd; ++target)
		if (target->name == name)
			return static_cast<uint4>(target - targets);

	return static_cast<uint4>(-1);
}

/// Gets all render targets in the given effect.
PipeEffectBinder::target_vector GetTargets(ID3DX11Effect *pEffect)
{
	PipeEffectBinder::target_vector targets;

	D3DX11_EFFECT_DESC effectDesc;
	BE_THROW_DX_ERROR_MSG(
		pEffect->GetDesc(&effectDesc),
		"ID3DX11Effect::GetDesc()");

	// Load targets
	for (UINT variableID = 0; variableID < effectDesc.GlobalVariables; ++variableID)
	{
		PipeEffectBinder::Target target;

		target.pTexture = pEffect->GetVariableByIndex(variableID)->AsShaderResource();

		// Skip non-samplers
		if (!target.pTexture->IsValid())
			continue;

		const char *targetTypeName = "";
		target.pTexture->GetAnnotationByName("TargetType")->AsString()->GetString(&targetTypeName);
		target.type = GetTargetType(targetTypeName);

		// Skip non-targets
		if (target.type == TargetType::Unknown)
			continue;

		D3DX11_EFFECT_VARIABLE_DESC variableDesc;
		BE_THROW_DX_ERROR_MSG(
			target.pTexture->GetDesc(&variableDesc),
			"ID3DX11EffectVariable::GetDesc()");

		// Skip unmanaged
		if (variableDesc.Flags & D3DX11_EFFECT_VARIABLE_UNMANAGED)
			continue;

		// Name required
		if (!variableDesc.Semantic || lean::char_traits<char>::empty(variableDesc.Semantic))
			LEAN_THROW_ERROR_CTX("GetTargets()", "Target missing name semantic.");

		target.name = variableDesc.Semantic;

		target.pTextureResolution = MaybeGetVectorVariable( pEffect, (target.name + "Resolution").c_str() );
		target.pTextureScaling = MaybeGetVectorVariable( pEffect, (target.name + "Scaling").c_str() );
		target.pTextureMultisampling = MaybeGetScalarVariable( pEffect, (target.name + "Multisampling").c_str() );

		BOOL bPerObject = false;
		target.pTexture->GetAnnotationByName("PerObject")->AsScalar()->GetBool(&bPerObject);
		target.bPerObject = (bPerObject != false);

		BOOL bOutput = false;
		target.pTexture->GetAnnotationByName("Output")->AsScalar()->GetBool(&bOutput);
		target.bOutput = (bOutput != false);

		const char *formatName = "R8G8B8A8U";
		target.pTexture->GetAnnotationByName("Format")->AsString()->GetString(&formatName);
		target.format = GetTargetFormat(formatName);

		// Valid format required
		if (target.format == DXGI_FORMAT_UNKNOWN)
			LEAN_THROW_ERROR_XCTX("GetTargetFormat()", "Target format unknown.", formatName);
		
		int mipLevels = 1;
		target.pTexture->GetAnnotationByName("MipLevels")->AsScalar()->GetInt(&mipLevels);
		target.MipLevels = static_cast<uint4>(mipLevels);

		BOOL bAutoGenMips = true;
		target.pTexture->GetAnnotationByName("AutoGenMips")->AsScalar()->GetBool(&bAutoGenMips);
		if (bAutoGenMips && target.MipLevels != 1)
			target.MipLevels |= beGraphics::TextureTargetFlags::AutoGenMipMaps;

		D3DX11_EFFECT_TYPE_DESC typeDesc;
		BE_THROW_DX_ERROR_MSG(
			target.pTexture->GetType()->GetDesc(&typeDesc),
			"ID3DX11EffectType::GetDesc()");
		target.bMultisampled = (typeDesc.Type == D3D_SVT_TEXTURE2DMS || typeDesc.Type == D3D_SVT_TEXTURE2DMSARRAY);

		// IMPORTANT: No source yet
		target.sourceTargetID = static_cast<uint4>(-1);

		targets.push_back(target);
	}

	// Link source targets
	for (PipeEffectBinder::target_vector::iterator itTarget = targets.begin(); itTarget != targets.end(); ++itTarget)
	{
		const char *sourceName = nullptr;
		itTarget->pTexture->GetAnnotationByName("Source")->AsString()->GetString(&sourceName);

		if (sourceName)
			itTarget->sourceTargetID = GetTargetIndex(targets.begin(), targets.end(), sourceName);
	}

	return targets;
}

/// Gets a color destinations from the given technique.
ColorDestination GetColorDestination(ID3DX11EffectPass *pPass, uint4 index,
	const PipeEffectBinder::Target *targets, const PipeEffectBinder::Target *targetsEnd)
{
	ColorDestination dest;

	const utf8_string indexPostfix =  lean::int_to_string(index);

	const char *destinationName = "";
	pPass->GetAnnotationByName( ("Color" + indexPostfix).c_str() )->AsString()->GetString(&destinationName);
	dest.targetID = GetTargetIndex(targets, targetsEnd, destinationName);

	BOOL bKeep = FALSE;
	pPass->GetAnnotationByName( ("bKeepColor" + indexPostfix).c_str() )->AsScalar()->GetBool(&bKeep);
	dest.bKeep = (bKeep != FALSE);

	BOOL bClearColor = FALSE;
	pPass->GetAnnotationByName( ("bClearColor" + indexPostfix).c_str() )->AsScalar()->GetBool(&bClearColor);
	dest.bClearColor = (bClearColor != FALSE);

	BOOL bClearColorOnce = FALSE;
	pPass->GetAnnotationByName( ("bClearColorOnce" + indexPostfix).c_str() )->AsScalar()->GetBool(&bClearColorOnce);
	dest.bClearColorOnce = (bClearColorOnce != FALSE);

	memset(dest.clearColor, 0, sizeof(dest.clearColor));
	pPass->GetAnnotationByName( ("ClearColor" + indexPostfix).c_str()  )->AsVector()->GetRawValue(dest.clearColor, 0, sizeof(dest.clearColor));

	return dest;
}

/// Gets a color destinations from the given pass.
DepthStencilDestination GetDepthStencilDestination(ID3DX11EffectPass *pPass,
	const PipeEffectBinder::Target *targets, const PipeEffectBinder::Target *targetsEnd)
{
	DepthStencilDestination dest;

	const char *destinationName = "";
	pPass->GetAnnotationByName("DepthStencil")->AsString()->GetString(&destinationName);
	dest.targetID = GetTargetIndex(targets, targetsEnd, destinationName);

	BOOL bKeep = false;
	pPass->GetAnnotationByName("bKeepDepthStencil")->AsScalar()->GetBool(&bKeep);
	dest.bKeep = (bKeep != false);

	dest.clearFlags = 0;

	BOOL bClearDepth = FALSE;
	pPass->GetAnnotationByName("bClearDepth")->AsScalar()->GetBool(&bClearDepth);
	if (bClearDepth)
		dest.clearFlags |= D3D11_CLEAR_DEPTH;
	
	BOOL bClearStencil = FALSE;
	pPass->GetAnnotationByName("bClearStencil")->AsScalar()->GetBool(&bClearStencil);
	if (bClearStencil)
		dest.clearFlags |= D3D11_CLEAR_STENCIL;

	BOOL bClearDepthOnce = FALSE;
	pPass->GetAnnotationByName("bClearDepthOnce")->AsScalar()->GetBool(&bClearDepthOnce);
	if (bClearDepthOnce)
		dest.clearOnceFlags |= D3D11_CLEAR_DEPTH;
	
	BOOL bClearStencilOnce = FALSE;
	pPass->GetAnnotationByName("bClearStencilOnce")->AsScalar()->GetBool(&bClearStencilOnce);
	if (bClearStencilOnce)
		dest.clearOnceFlags |= D3D11_CLEAR_STENCIL;

	dest.clearDepth = 1.0f;
	pPass->GetAnnotationByName("ClearDepth")->AsScalar()->GetFloat(&dest.clearDepth);

	int clearStencil = 0;
	pPass->GetAnnotationByName("ClearStencil")->AsScalar()->GetInt(&clearStencil);
	dest.clearStencil = static_cast<uint4>(clearStencil);

	return dest;
}

/// Gets the given pass from the given technique.
PipeEffectBinder::Pass GetPass(ID3DX11EffectTechnique *pTechnique, UINT passID,
	PipeEffectBinder::Target *targets, PipeEffectBinder::Target *targetsEnd,
	uint4 binderFlags)
{
	PipeEffectBinder::Pass pass(
		passID,
		pTechnique->GetPassByIndex(passID));

	if (!pass.pPass->IsValid())
		LEAN_THROW_ERROR_MSG("ID3DX11Technique::GetPassByIndex()");

	pass.colorCount = 0;

	while (pass.colorCount < MaxSimultaneousDestinationCount)
	{
		ColorDestination dest = GetColorDestination(pass.pPass, pass.colorCount, targets, targetsEnd);

		if (dest.targetID == static_cast<uint4>(-1))
			break;
		
		pass.color[pass.colorCount] = dest;
		++pass.colorCount;
	}

	pass.depthStencil = GetDepthStencilDestination(pass.pPass, targets, targetsEnd);

	pass.scaleX = 0.0f;
	pass.pPass->GetAnnotationByName("ScaleX")->AsScalar()->GetFloat(&pass.scaleX);
	pass.scaleY = 0.0f;
	pass.pPass->GetAnnotationByName("ScaleY")->AsScalar()->GetFloat(&pass.scaleY);

	int resolutionX = 0;
	pass.pPass->GetAnnotationByName("ResolutionX")->AsScalar()->GetInt(&resolutionX);
	pass.resolutionX = static_cast<uint4>(resolutionX);
	int resolutionY = 0;
	pass.pPass->GetAnnotationByName("ResolutionY")->AsScalar()->GetInt(&resolutionY);
	pass.resolutionY = static_cast<uint4>(resolutionY);

	BOOL multisamped = (binderFlags & PipeEffectBinderFlags::NoDefaultMS) ? FALSE : TRUE;
	pass.pPass->GetAnnotationByName("Multisampled")->AsScalar()->GetBool(&multisamped);
	pass.bMultisampled = (multisamped != FALSE);

	BOOL multisamplingOnly = FALSE;
	pass.pPass->GetAnnotationByName("MultisamplingOnly")->AsScalar()->GetBool(&multisamplingOnly);
	pass.bMultisamplingOnly = (multisamplingOnly != FALSE);

	// NOTE: Required to resolve resource hazards in time
	BOOL revertTargets = FALSE;
	pass.pPass->GetAnnotationByName("RevertTargets")->AsScalar()->GetBool(&revertTargets);
	pass.bRevertTargets = (revertTargets != FALSE);

	return pass;
}

/// Gets all passes in the given technique.
PipeEffectBinder::pass_vector GetPasses(ID3DX11EffectTechnique *pTechnique,
	PipeEffectBinder::Target *targets, PipeEffectBinder::Target *targetsEnd,
	uint4 binderFlags,
	uint4 singlePassID = static_cast<uint4>(-1))
{
	PipeEffectBinder::pass_vector passes;

	D3DX11_TECHNIQUE_DESC techniqueDesc;
	BE_THROW_DX_ERROR_MSG(
		pTechnique->GetDesc(&techniqueDesc),
		"ID3DX11Technique::GetDesc()");
	
	if (singlePassID < techniqueDesc.Passes)
		// Load single pass
		passes.push_back( GetPass(pTechnique, singlePassID, targets, targetsEnd, binderFlags) );
	else
	{
		passes.reserve(techniqueDesc.Passes);

		// Load all passes
		for (UINT passID = 0; passID < techniqueDesc.Passes; ++passID)
			passes.push_back( GetPass(pTechnique, passID, targets, targetsEnd, binderFlags) );
	}

	for (PipeEffectBinder::Target *target = targets; target != targetsEnd; ++target)
	{
		uint4 targetID = static_cast<uint4>(target - targets);

		if (target->type == TargetType::Temporary)
		{
			// Dispose immediately, if unused
			target->disposePassID = 0;

			// Dispose as soon as possible
			for (PipeEffectBinder::pass_vector::const_reverse_iterator itPass = passes.rbegin();
				target->disposePassID == 0 && itPass != passes.rend(); ++itPass)
			{
				// DON'T check color writes, rendering to color targets is write-only
//				for (uint4 i = 0; i < itPass->colorCount; ++i)
//					if (itPass->color[i].targetID == targetID)
//						target->disposePassID = itPass.base() - passes.begin();

				if (itPass->depthStencil.targetID == targetID)
					target->disposePassID = static_cast<uint4>(itPass.base() - passes.begin());
			}
		}
		else
			// Don't dispose
			target->disposePassID = static_cast<uint4>(-1);
	}

	return passes;
}

/// Gets usage flags for the given pass & target combinations.
PipeEffectBinder::used_bitset GetTargetsUsedFlags(ID3DX11Effect *pEffect,
	const PipeEffectBinder::Pass *passes, const PipeEffectBinder::Pass *passesEnd,
	PipeEffectBinder::Target *targets, PipeEffectBinder::Target *targetsEnd)
{
	const size_t passCount = passesEnd - passes;
	const size_t targetCount = targetsEnd - targets;

	PipeEffectBinder::used_bitset targetsUsed(passCount * targetCount);
	
	for (uint4 passID = 0; passID < passCount; ++passID)
	{
		const PipeEffectBinder::Pass &pass = passes[passID];

		size_t targetsUsedBase = passID * targetCount;

		beGraphics::Any::API::EffectString *pForceBinding = pass.pPass->GetAnnotationByName("ForceTextureBinding")->AsString();

		// Forced usage
		if (pForceBinding->IsValid())
		{
			const uint4 bindingCount = beGraphics::GetDesc(pForceBinding->GetType()).Elements;
			
			for (uint4 targetID = 0; targetID < targetCount; ++targetID)
			{
				const PipeEffectBinder::Target &target = targets[targetID];
				D3DX11_EFFECT_VARIABLE_DESC variableDesc = beGraphics::GetDesc(target.pTexture);

				for (uint4 i = 0; i < bindingCount; ++i)
				{
					const char *targetName = "";
					pForceBinding->GetStringArray(&targetName, i, 1);

					if (lean::char_traits<char>::equal(variableDesc.Name, targetName))
						targetsUsed[targetsUsedBase + targetID] = true;
				}
			}
		}

		// Track usage in shaders
		lean::com_ptr<ID3D11ShaderReflection> pReflections[beGraphics::Any::ShaderType::End];

		for (int i = 0; i < beGraphics::Any::ShaderType::End; ++i)
			pReflections[i] = beGraphics::Any::MaybeReflectShader(pass.pPass, static_cast<beGraphics::Any::ShaderType::T>(i));

		for (uint4 targetID = 0; targetID < targetCount; ++targetID)
		{
			const PipeEffectBinder::Target &target = targets[targetID];
			D3DX11_EFFECT_VARIABLE_DESC variableDesc = beGraphics::GetDesc(target.pTexture);

			for (int i = 0; i < beGraphics::Any::ShaderType::End; ++i)
			{
				if (pReflections[i])
				{
					D3D11_SHADER_INPUT_BIND_DESC shaderResourceBindingDesc;

					if ( SUCCEEDED(pReflections[i]->GetResourceBindingDescByName(variableDesc.Name, &shaderResourceBindingDesc)) )
						targetsUsed[targetsUsedBase + targetID] = true;
				}
			}
		}
	}

	for (uint4 targetID = 0; targetID < targetCount; ++targetID)
	{
		PipeEffectBinder::Target &target = targets[targetID];

		// Delay target disposal until target no longer read
		for (uint4 passID = static_cast<uint4>(passCount); target.disposePassID < passID && passID-- > 0; )
			if (targetsUsed[passID * targetCount + targetID])
				target.disposePassID = passID;
	}
	
	return targetsUsed;

}

} // namespace

// Constructor.
PipeEffectBinder::PipeEffectBinder(const beGraphics::Any::Technique &technique, uint4 flags, uint4 passID)
	: m_technique( technique ),
	m_targets( GetTargets(*m_technique.GetEffect()) ),
	m_passes( GetPasses(m_technique, &m_targets[0], &m_targets[0] + m_targets.size(), flags, passID) ),

	m_targetsUsedPitch(static_cast<uint4>(m_targets.size())),
	m_targetsUsed(
		GetTargetsUsedFlags(*m_technique.GetEffect(),
			&m_passes[0], &m_passes[0] + m_passes.size(),
			&m_targets[0], &m_targets[0] + m_targets.size()) ),

	m_pResolution( MaybeGetVectorVariable(*m_technique.GetEffect(), "Resolution") ),
	m_pMultisampling( MaybeGetScalarVariable(*m_technique.GetEffect(), "Multisampling") ),

	m_pDestinationScaling( MaybeGetVectorVariable(*m_technique.GetEffect(), "DestinationScaling") ),
	m_pDestinationResolution( MaybeGetVectorVariable(*m_technique.GetEffect(), "DestinationResolution") ),
	m_pDestinationMultisampling( MaybeGetScalarVariable(*m_technique.GetEffect(), "DestinationMultisampling") )
{
}

// Destructor.
PipeEffectBinder::~PipeEffectBinder()
{
}

namespace
{

/// Generates an object-specific name from the given name in the given character buffer.
/// ASSUMES the buffer is long enough to hold all decimal representations of pointer-sized integers.
LEAN_INLINE utf8_ntr GenerateLocalName(utf8_t *buffer, size_t bufferLen, const utf8_ntri &name, const void *pObject)
{
	LEAN_ASSERT(bufferLen > lean::max_int_string_length<intptr_t>::value);

	utf8_t *numEnd = lean::int_to_char(buffer, reinterpret_cast<uintptr_t>(pObject));

	return utf8_ntr(
		buffer,
		numEnd + lean::strmcpy( numEnd, name.data(), bufferLen - (numEnd - buffer) ) );
}

/// Gets the target name.
template <size_t BufferLen>
inline utf8_ntr GetTargetName(const PipeEffectBinder::Target &target, utf8_t (&nameBuffer)[BufferLen], const void *pObject)
{
	// Build object-specific name, if requested
	return (target.bPerObject)
		? GenerateLocalName(nameBuffer, lean::arraylen(nameBuffer), target.name, pObject)
		: utf8_ntr(target.name);
}

/// Sets the sample count, if requested.
LEAN_INLINE void MaybeSetMultisampling(ID3DX11EffectScalarVariable *pVariable, const beGraphics::Any::TextureTargetDesc &desc)
{
	if (pVariable)
	{
		int sampleCount = desc.Samples.Count;
		pVariable->SetInt(sampleCount);
	}
}

/// Sets the texture target resolution, if requested.
LEAN_INLINE void MaybeSetResolution(ID3DX11EffectVectorVariable *pVariable, const beGraphics::Any::TextureTargetDesc &desc)
{
	if (pVariable)
	{
		float resolution[4] = {
			(float) desc.Width, (float) desc.Height,
			1.0f / desc.Width, 1.0f / desc.Height };
		pVariable->SetFloatVector(resolution);
	}
}

/// Sets the texture target scaling, if requested.
LEAN_INLINE void MaybeSetScaling(ID3DX11EffectVectorVariable *pVariable,
	const beGraphics::Any::TextureTargetDesc &desc, const beGraphics::Any::TextureTargetDesc &frameDesc)
{
	if (pVariable)
	{
		float scaling[4] = {
			(float) desc.Width / frameDesc.Width,
			(float) desc.Height / frameDesc.Height,
			(float) frameDesc.Width / desc.Width,
			(float) frameDesc.Height / desc.Height };
		pVariable->SetFloatVector(scaling);
	}
}

/// Constructs a default viewport from the given target texture description.
LEAN_INLINE D3D11_VIEWPORT ToViewport(const beGraphics::Any::TextureTargetDesc &desc)
{
	D3D11_VIEWPORT viewport;
	viewport.TopLeftX = 0.0f;
	viewport.TopLeftY = 0.0f;
	viewport.Width = (float) desc.Width;
	viewport.Height = (float) desc.Height;
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;
	return viewport;
}

/// Converts the given target type to pipe target flags.
LEAN_INLINE uint4 ToTargetFlags(TargetType::T type)
{
	return (type == TargetType::Persistent)
		? PipeTargetFlags::Persistent
		: 0;
}

/// Sets the given target texture in the given device context.
LEAN_INLINE void SetTargetTexture(const PipeEffectBinder::Target &target, const PipeEffectBinder::Target *targets, uint4 targetCount,
	DX11::Pipe *pPipe, uint4 outputIndex, const void *pObject,
	ID3D11DeviceContext *pContext)
{
	utf8_t nameBuffer[256];
	utf8_ntr targetName = GetTargetName(target, nameBuffer, pObject);

	const beGraphics::TextureTarget *pTextureTarget = pPipe->GetColorTarget(targetName);
	bool bColorTarget = (pTextureTarget != nullptr);

	// Try depth-stencil target on color miss
	if (!bColorTarget)
		pTextureTarget = pPipe->GetDepthStencilTarget(targetName);

	const PipeEffectBinder::Target *pSourceTarget = &target;
	utf8_t sourceNameBuffer[256];
	utf8_ntr sourceTargetName = targetName;

	uint4 sourceLoopCounter = 0;

	// Redirect to source target, if actual target unavailable and source target specified
	while (!pTextureTarget && pSourceTarget->sourceTargetID != static_cast<uint4>(-1) &&
			// Stay clear of endless loops
			++sourceLoopCounter < targetCount)
	{
		pSourceTarget = &targets[pSourceTarget->sourceTargetID];
		sourceTargetName = GetTargetName(*pSourceTarget, sourceNameBuffer, pObject);

		pTextureTarget = pPipe->GetColorTarget(sourceTargetName);
		bColorTarget = (pTextureTarget != nullptr);

		// Try depth-stencil target on color miss
		if (!bColorTarget)
			pTextureTarget = pPipe->GetDepthStencilTarget(sourceTargetName);
	}

	// Always log essential errors
	if (sourceLoopCounter == targetCount)
		LEAN_LOG_ERROR_CTX("Non-terminating texture target source cycle detected!", targetName.c_str());

	if (pTextureTarget)
	{
		// Resolve multisampled texture, if necessary
		if (pTextureTarget->GetDesc().Samples.Count > 1 && !target.bMultisampled)
		{
			lean::com_ptr<ID3D11Resource> pUnresolved = pTextureTarget->GetResource();

			beGraphics::Any::TextureTargetDesc resolvedDesc = pTextureTarget->GetDesc();
			resolvedDesc.Samples.Count = 1;
			resolvedDesc.Samples.Quality = 0;
			resolvedDesc.MipLevels = target.MipLevels;

			uint4 targetFlags = ToTargetFlags(target.type);

			const beGraphics::TextureTarget *pResolvedTextureTarget;

			// Replace original multisampled target by new non-multisampled resolvation target texture
			if (bColorTarget)
				pResolvedTextureTarget = pPipe->GetNewColorTarget(targetName, resolvedDesc, targetFlags | PipeTargetFlags::Keep, outputIndex);
			else
				pResolvedTextureTarget = pPipe->GetNewDepthStencilTarget(targetName, resolvedDesc, targetFlags | PipeTargetFlags::Keep, outputIndex);

			// Resolve original multisampled target to new non-multisampled target texture
			if (pResolvedTextureTarget->GetTexture())
			{
				pContext->ResolveSubresource(pResolvedTextureTarget->GetResource(), 0, pUnresolved, 0, resolvedDesc.Format);
				pTextureTarget = pResolvedTextureTarget;
			}
			// Always log essential errors
			else
				LEAN_LOG_ERROR_CTX("Multisampled target cannot be resolved!", sourceTargetName.c_str());
		}

		// Re-generate mip levels before rendering
		if (target.MipLevels & beGraphics::TextureTargetFlags::AutoGenMipMaps)
			pContext->GenerateMips(pTextureTarget->GetTexture());

		target.pTexture->SetResource(pTextureTarget->GetTexture());

		// Always log essential errors
		if (!pTextureTarget->GetTexture())
			LEAN_LOG_ERROR_CTX("Target cannot be bound as texture!", sourceTargetName.c_str());
		
		MaybeSetResolution(target.pTextureResolution, pTextureTarget->GetDesc());
		MaybeSetScaling(target.pTextureScaling, pTextureTarget->GetDesc(), pPipe->GetDesc());
		MaybeSetMultisampling(target.pTextureMultisampling, pTextureTarget->GetDesc());
	}
	else
		// Unbind texture, if unavailable
		target.pTexture->SetResource(nullptr);
}

/// Sets the targets of the given pass in the given context. Returns true to indicate that the current pass should be repeated.
LEAN_INLINE bool SetTargets(const PipeEffectBinder::Pass &pass, const PipeEffectBinder::Target *targets,
	ID3DX11EffectVectorVariable *pResolution, ID3DX11EffectVectorVariable *pScaling, ID3DX11EffectScalarVariable *pMultisampling,
	DX11::Pipe *pPipe, uint4 outputIndex, const void *pObject,
	beGraphics::Any::StateManager& stateManager, ID3D11DeviceContext *pContext)
{
	bool bRepeatPass = false;

	bool bHasColorTarget = (pass.colorCount != 0);
	bool bHasDepthStencilTarget = (pass.depthStencil.targetID != static_cast<uint4>(-1));

	if (bHasColorTarget || bHasDepthStencilTarget)
	{
		beGraphics::Any::TextureTargetDesc mainDesc = pPipe->GetDesc();

		// Disable multisampling, if requested
		if (!pass.bMultisampled)
		{
			mainDesc.Samples.Count = 1;
			mainDesc.Samples.Quality = 0;
		}

		bool bHasResolutionX = (pass.resolutionX != 0);
		bool bHasResolutionY = (pass.resolutionY != 0);

		// Set fixed resolution, if requested
		if (bHasResolutionX)
			mainDesc.Width = pass.resolutionX;
		if (bHasResolutionY)
			mainDesc.Height = pass.resolutionY;

		bool bHasScalingX = (pass.scaleX > 0.0f);
		bool bHasScalingY = (pass.scaleY > 0.0f);

		// Perform dynamic scaling, if requested
		if (bHasScalingX || bHasScalingY)
		{
			const beGraphics::TextureTarget *pMainTarget = nullptr;

			// Find reference target, if scaling requested
			for (uint4 i = 0; !pMainTarget && i < pass.colorCount; ++i)
			{
				const ColorDestination &dest = pass.color[i];

				const PipeEffectBinder::Target &target = targets[dest.targetID];
				utf8_t nameBuffer[256];
				utf8_ntr targetName = GetTargetName(target, nameBuffer, pObject);

				pMainTarget = pPipe->GetColorTarget(targetName);
			}

			if (!pMainTarget && bHasDepthStencilTarget)
			{
				const DepthStencilDestination &dest = pass.depthStencil;
				
				const PipeEffectBinder::Target &target = targets[dest.targetID];
				utf8_t nameBuffer[256];
				utf8_ntr targetName = GetTargetName(target, nameBuffer, pObject);

				pMainTarget = pPipe->GetDepthStencilTarget(targetName);
			}

			// Use pipe description, if no reference target available.
			if (pMainTarget)
			{
				mainDesc.Width = pMainTarget->GetDesc().Width;
				mainDesc.Height = pMainTarget->GetDesc().Height;
			}

			if (bHasScalingX)
			{
				mainDesc.Width = static_cast<uint4>(mainDesc.Width * pass.scaleX);

				if (bHasResolutionX)
				{
					// Check if scaling still in range
					bHasScalingX = (pass.scaleX < 1.0f) && (mainDesc.Width > pass.resolutionX)
						|| (pass.scaleX > 1.0f) && (mainDesc.Width < pass.resolutionX);

					// Repeat pass until target resolution reached
					if (bHasScalingX)
						bRepeatPass |= true;
					else
						mainDesc.Width = pass.resolutionX;
				}
			}
			if (bHasScalingY)
			{
				mainDesc.Height = static_cast<uint4>(mainDesc.Height * pass.scaleY);

				if (bHasResolutionY)
				{
					// Check if scaling still in range
					bHasScalingY = (pass.scaleY < 1.0f) && (mainDesc.Height > pass.resolutionY)
						|| (pass.scaleY > 1.0f) && (mainDesc.Height < pass.resolutionY);

					// Repeat pass until target resolution reached
					if (bHasScalingY)
						bRepeatPass |= true;
					else
						mainDesc.Height = pass.resolutionY;
				}
			}
		}

		// NOTE: Temporarily store references to all overwritten targets to prevent them from being re-used while still bound as texture
		lean::com_ptr<const beGraphics::ColorTextureTarget> colorReferenceStore[MaxSimultaneousDestinationCount];
		lean::com_ptr<const beGraphics::DepthStencilTextureTarget> depthStencilReferenceStore;

		ID3D11RenderTargetView *renderTargets[MaxSimultaneousDestinationCount] = { nullptr };

		for (uint4 i = 0; i < pass.colorCount; ++i)
		{
			const ColorDestination &dest = pass.color[i];

			const PipeEffectBinder::Target &target = targets[dest.targetID];
			utf8_t nameBuffer[256];
			utf8_ntr targetName = GetTargetName(target, nameBuffer, pObject);

			beGraphics::Any::TextureTargetDesc targetDesc = mainDesc;

			// Override format, if specified
			if (target.format != DXGI_FORMAT_UNKNOWN)
				targetDesc.Format = target.format;
			targetDesc.MipLevels = target.MipLevels;

			uint4 targetFlags = ToTargetFlags(target.type);
			if (target.bOutput)
				targetFlags |= PipeTargetFlags::Output;

			bool bTargetNew = !dest.bKeep;

			// Only keep target if requested
			const beGraphics::ColorTextureTarget *pTextureTarget = (dest.bKeep)
				? pPipe->GetColorTarget( targetName, targetDesc, targetFlags, outputIndex, &bTargetNew)
				: pPipe->GetNewColorTarget( targetName, targetDesc, targetFlags, outputIndex, &colorReferenceStore[i]);

			if (pTextureTarget)
			{
				ID3D11RenderTargetView *pRenderTarget = (target.bOutput)
					? pTextureTarget->GetTarget(outputIndex)
					: pTextureTarget->GetTarget();

				if (dest.bClearColor | bTargetNew & dest.bClearColorOnce)
					pContext->ClearRenderTargetView(pRenderTarget, dest.clearColor);

				renderTargets[i] = pRenderTarget;
			}
			// Always log essential errors
			else
				LEAN_LOG_ERROR_CTX("Color target cannot be rendered to!", targetName.c_str());
		}

		ID3D11DepthStencilView *pDepthStencilTarget = nullptr;

		if (bHasDepthStencilTarget)
		{
			const DepthStencilDestination &dest = pass.depthStencil;
			
			const PipeEffectBinder::Target &target = targets[dest.targetID];
			utf8_t nameBuffer[256];
			utf8_ntr targetName = GetTargetName(target, nameBuffer, pObject);

			beGraphics::Any::TextureTargetDesc targetDesc = mainDesc;

			// Only keep target if requested
			if (target.format != DXGI_FORMAT_UNKNOWN)
				targetDesc.Format = target.format;

			uint4 targetFlags = ToTargetFlags(target.type);
			if (target.bOutput)
				targetFlags |= PipeTargetFlags::Output;

			bool bTargetNew = !dest.bKeep;

			const beGraphics::DepthStencilTextureTarget *pTextureTarget = (dest.bKeep)
				? pPipe->GetDepthStencilTarget( targetName, targetDesc, targetFlags, outputIndex, &bTargetNew)
				: pPipe->GetNewDepthStencilTarget( targetName, targetDesc, targetFlags, outputIndex, &depthStencilReferenceStore);

			// Only keep target, if requested
			if (pTextureTarget)
			{
				uint4 clearFlags = dest.clearFlags;

				if (bTargetNew)
					clearFlags |= dest.clearOnceFlags;

				pDepthStencilTarget = (target.bOutput)
					? pTextureTarget->GetTarget(outputIndex)
					: pTextureTarget->GetTarget();

				if (clearFlags)
					pContext->ClearDepthStencilView(pDepthStencilTarget,
						clearFlags, dest.clearDepth, dest.clearStencil);
			}
			// Always log essential errors
			else
				LEAN_LOG_ERROR_CTX("Depth-stencil target cannot be rendered to!", targetName.c_str());
		}
		// DON'T DO THIS:
		// -> Allows for un-buffered rendering
		// -> Prevents accidental format mismatches
//		else
//			pContext->OMGetRenderTargets(0, nullptr, &pDepthStencilTarget);

		stateManager.Override(beGraphics::DX11::StateMasks::RenderTargets);

		pContext->OMSetRenderTargetsAndUnorderedAccessViews(pass.colorCount, renderTargets,
			pDepthStencilTarget,
			pass.colorCount, D3D11_KEEP_UNORDERED_ACCESS_VIEWS, nullptr, nullptr);

		pContext->RSSetViewports(1, &ToViewport(mainDesc));

		MaybeSetResolution(pResolution, mainDesc);
		MaybeSetScaling(pScaling, mainDesc, pPipe->GetDesc());
		MaybeSetMultisampling(pMultisampling, mainDesc);
	}
	// IMPORTANT: Resolve resource hazards early
	else if (pass.bRevertTargets)
		pContext->OMSetRenderTargetsAndUnorderedAccessViews(0, nullptr, nullptr,
			0, D3D11_KEEP_UNORDERED_ACCESS_VIEWS, nullptr, nullptr);

	return bRepeatPass;
}

} // namespace

// Applies the n-th step of the given pass.
bool PipeEffectBinder::Apply(uint4 &nextPassID, DX11::Pipe *pPipe, uint4 outputIndex, const void *pObject,
	beGraphics::Any::StateManager& stateManager, ID3D11DeviceContext *pContext) const
{
	uint4 passID = nextPassID++;

	if (passID >= m_passes.size())
		return false;

	const Pass &pass = m_passes[passID];

	if (pPipe)
	{
		const uint4 targetCount = static_cast<uint4>(m_targets.size());

		// ORDER: Set target textures BEFORE being overwritten by this pass
		for (uint4 targetID = 0; targetID < targetCount; ++targetID)
			if (m_targetsUsed[passID * m_targetsUsedPitch + targetID])
				SetTargetTexture(m_targets[targetID], &m_targets[0], targetCount, pPipe, outputIndex, pObject, pContext);

		// ORDER: Only replace texture targets AFTER having been set by this pass
		if ( SetTargets(pass, &m_targets[0],
				m_pDestinationResolution, m_pDestinationScaling, m_pDestinationMultisampling,
				pPipe, outputIndex, pObject, stateManager, pContext) )
			// Repeat pass, if requested
			--nextPassID;

		MaybeSetResolution(m_pResolution, pPipe->GetDesc());
		MaybeSetMultisampling(m_pMultisampling, pPipe->GetDesc());

		// ORDER: Dispose after everything has been set
		for (target_vector::const_iterator itTarget = m_targets.begin(); itTarget != m_targets.end(); ++itTarget)
		{
			const Target &target = *itTarget;

			// Dispose temporary targets as soon as possible
			if (target.disposePassID <= passID)
			{
				utf8_t nameBuffer[256];
				utf8_ntr targetName = GetTargetName(target, nameBuffer, pObject);

				pPipe->SetColorTarget(targetName, nullptr, outputIndex, 0);
				pPipe->SetDepthStencilTarget(targetName, nullptr, outputIndex, 0);
			}
		}
	}

	return true;
}

} // namespace