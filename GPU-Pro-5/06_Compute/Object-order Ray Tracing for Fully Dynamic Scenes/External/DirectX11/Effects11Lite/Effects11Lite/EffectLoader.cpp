//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "D3D11EffectsLite.h"
#include "Reflection.h"
#include "Allocator.h"
#include "Errors.h"
#include "StringHelper.h"
#include <cassert>

#include "Binary/EffectBinaryFormat.h"

namespace D3DEffectsLite
{

namespace
{

const UINT ConstantBufferAlignment = sizeof(FLOAT) * 4;

struct invalidptr_t
{
	template <class T>
	operator T*() const { return reinterpret_cast<T*>( UINT_PTR(-1) ); }
} const invalidptr;

template <class T>
bool ptrvalid(T *p)
{
	return (p != nullptr) && (p != invalidptr);
}

struct ReaderUnderflow { };
struct UnknownVersionError { };
struct CheckError { const char *msg; CheckError(const char *msg) : msg(msg) { assert(msg); } };

void Check(bool b, const char *msg)
{
	if (!b)
		throw CheckError(msg);
}

struct Reader
{
	const BYTE *base;
	UINT size;
	UINT pointer;

	Reader(const void *base, UINT size, UINT pointer = 0)
		: base( static_cast<const BYTE*>(base) ),
		size( size ),
		pointer( pointer ) { }
};

void CheckFetchSize(const Reader& reader, UINT at, UINT size)
{
	if (at + size > reader.size)
		throw ReaderUnderflow();
}

template <class T>
const T& Fetch(Reader& reader, UINT at)
{
	CheckFetchSize(reader, at, sizeof(T));
	return *reinterpret_cast<const T*>(reader.base + at);
}

template <class T>
const T* FetchMultiple(Reader& reader, UINT at, UINT count)
{
	CheckFetchSize(reader, at, count * sizeof(T));
	return reinterpret_cast<const T*>(reader.base + at);
}

template <class T>
const T& Peek(Reader& reader)
{
	return Fetch<T>(reader, reader.pointer);
}

template <class T>
const T* PeekMultiple(Reader& reader, UINT count)
{
	return FetchMultiple<T>(reader, reader.pointer, count);
}

template <class T>
const T& Read(Reader& reader)
{
	const T &v = Peek<T>(reader);
	reader.pointer += sizeof(T);
	return v;
}

template <class T>
const T* ReadMultiple(Reader& reader, UINT count)
{
	const T *v = PeekMultiple<T>(reader, count);
	reader.pointer += count * sizeof(T);
	return v;
}

DWORD GetEffectVersion(UINT effectFileTag)
{
	for (UINT i = 0; i < arraylen(D3DX11Effects::g_EffectVersions); ++i)
		if (D3DX11Effects::g_EffectVersions[i].m_Tag == effectFileTag)
			return D3DX11Effects::g_EffectVersions[i].m_Version;

	throw UnknownVersionError();
}

template <class T, class C, class A>
struct ArrayPairBase
{
	C Count;
	A Array;

	ArrayPairBase(C count, A array)
		: Count(count),
		Array(array) { }
};

template <class T>
struct ArrayPair : public ArrayPairBase<T, UINT, T*>
{
	ArrayPair(UINT count = 0, T* array = nullptr)
		: ArrayPairBase<T, UINT, T*>(count, array) { }
};

template <class T>
struct DestructArrayPair : public ArrayPair<T>
{
	DestructArrayPair(UINT count = 0, T* array = nullptr)
		: ArrayPair<T>(count, array) { }
	~DestructArrayPair()
	{
		if (this->Array)
			for (UINT i = 0; i < Count; ++i)
				this->Array[i].~T();
	}

private:
	DestructArrayPair(const DestructArrayPair &pair);
	DestructArrayPair& operator =(const DestructArrayPair &pair);
};

template <class T>
struct ScratchArrayPair : public ArrayPair<T>
{
	Allocator *Allocator;
	UINT AllocationCount;

	ScratchArrayPair(D3DEffectsLite::Allocator *allocator = nullptr, UINT count = 0, T* array = nullptr, UINT allocationCount = 0)
		: ArrayPair<T>(count, array),
		Allocator(allocator),
		AllocationCount(allocationCount) { }
	~ScratchArrayPair()
	{
		if (Allocator)
			AllocatorDeleteMultiple(*Allocator, this->Array, AllocationCount);
	}

private:
	ScratchArrayPair(const ScratchArrayPair &pair);
	ScratchArrayPair& operator =(const ScratchArrayPair &pair);
};

template <class T>
struct ArrayPairRef : public ArrayPairBase<T, UINT&, T*&>
{
	ArrayPairRef(UINT& count, T*& array)
		: typename ArrayPairRef::ArrayPairBase(count, array) { }
	ArrayPairRef(ArrayPairBase<T, UINT&, T*&> &pair)
		: typename ArrayPairRef::ArrayPairBase(pair.Count, pair.Array) { }

	ArrayPairRef& operator =(const ArrayPair<T> &pair)
	{
		Count = pair.Count;
		Array = pair.Array;
		return *this;
	}
};

template <class T>
ArrayPairRef<T> MakeRef(Heap &heap, ArrayPairBase<T, UINT&, T*&> &pair)
{
	return ArrayPairRef<T>(pair);
}

template <class T, class C, class A>
T* BasePointer(const ArrayPairBase<T, C, A> &pair)
{
	return (pair.Array)
		? pair.Array + pair.Count
		: nullptr;
}

template <class T, class C, class A>
ArrayPair<T> Range(const ArrayPairBase<T, C, A> &pair, T *base)
{
	UINT count = static_cast<UINT>(pair.Array + pair.Count - base);
	return ArrayPair<T>(count, (count > 0) ? base : nullptr);
}

template <class T>
void NoConstructMultipleNeverNull(Heap &heap, ArrayPairRef<T> pair)
{
	pair.Array = heap.NoConstructMultiple<T>(pair.Count);
}

template <class T>
void ConstructMultipleNeverNull(Heap &heap, ArrayPairRef<T> pair)
{
	pair.Array = heap.ConstructMultiple<T>(pair.Count);
}

template <class T>
void NoConstructMultiple(Heap &heap, ArrayPairRef<T> pair)
{
	if (pair.Count > 0)
		NoConstructMultipleNeverNull(heap, pair);
	else
		pair.Array = nullptr;
}

template <class T>
void ConstructMultiple(Heap &heap, ArrayPairRef<T> pair)
{
	if (pair.Count > 0)
		ConstructMultipleNeverNull(heap, pair);
	else
		pair.Array = nullptr;
}

template <class T>
T* AddAndMaybeConstruct(ArrayPairRef<T> pair)
{
	T *p = (pair.Array)
		? new(pair.Array + pair.Count) T()
		: nullptr;
	// NOTE: ALWAYS increment
	++pair.Count;
	return p;
}

template <class T>
T* AddAndMaybeConstructMultiple(ArrayPairRef<T> pair, UINT count)
{
	T *p = (pair.Array) ? pair.Array + pair.Count : nullptr;
	if (p)
		for (UINT i = 0; i < count; ++i)
			new(p + i) T();
	// NOTE: ALWAYS increment
	pair.Count += count;
	return p;
}

ArrayPairRef<VariableInfo> Variables(EffectInfo &info) { return ArrayPairRef<VariableInfo>(info.VariableCount, info.Variables.get()); }

ArrayPairRef<GroupInfo> Groups(EffectInfo &info) { return ArrayPairRef<GroupInfo>(info.GroupCount, info.Groups.get()); }
ArrayPairRef<TechniqueInfo> Techniques(EffectInfo &info) { return ArrayPairRef<TechniqueInfo>(info.TechniqueCount, info.Techniques.get()); }
ArrayPairRef<PassInfo> Passes(EffectInfo &info) { return ArrayPairRef<PassInfo>(info.PassCount, info.Passes.get()); }

ArrayPairRef<TechniqueInfo> Techniques(GroupInfo &info) { return ArrayPairRef<TechniqueInfo>(info.TechniqueCount, info.Techniques.get()); }
ArrayPairRef<PassInfo> Passes(TechniqueInfo &info) { return ArrayPairRef<PassInfo>(info.PassCount, info.Passes.get()); }

ArrayPairRef<FloatVariableInfo> Floats(EffectConstantInfo &info) { return ArrayPairRef<FloatVariableInfo>(info.FloatCount, info.Floats.get()); }
ArrayPairRef<IntVariableInfo> Ints(EffectConstantInfo &info) { return ArrayPairRef<IntVariableInfo>(info.IntCount, info.Ints.get()); }
ArrayPairRef<BoolVariableInfo> Bools(EffectConstantInfo &info) { return ArrayPairRef<BoolVariableInfo>(info.BoolCount, info.Bools.get()); }
ArrayPairRef<StructVariableInfo> Structs(EffectConstantInfo &info) { return ArrayPairRef<StructVariableInfo>(info.StructCount, info.Structs.get()); }
ArrayPairRef<StringVariableInfo> Strings(EffectConstantInfo &info) { return ArrayPairRef<StringVariableInfo>(info.StringCount, info.Strings.get()); }

ArrayPairRef<FloatAnnotationInfo> Floats(AnnotationBlockInfo &info) { return ArrayPairRef<FloatAnnotationInfo>(info.FloatCount, info.Floats.get()); }
ArrayPairRef<IntAnnotationInfo> Ints(AnnotationBlockInfo &info) { return ArrayPairRef<IntAnnotationInfo>(info.IntCount, info.Ints.get()); }
ArrayPairRef<BoolAnnotationInfo> Bools(AnnotationBlockInfo &info) { return ArrayPairRef<BoolAnnotationInfo>(info.BoolCount, info.Bools.get()); }
ArrayPairRef<StringAnnotationInfo> Strings(AnnotationBlockInfo &info) { return ArrayPairRef<StringAnnotationInfo>(info.StringCount, info.Strings.get()); }

ArrayPairRef<FloatVariableInfo> Floats(ConstantBufferInfo &info) { return ArrayPairRef<FloatVariableInfo>(info.FloatCount, info.Floats.get()); }
ArrayPairRef<IntVariableInfo> Ints(ConstantBufferInfo &info) { return ArrayPairRef<IntVariableInfo>(info.IntCount, info.Ints.get()); }
ArrayPairRef<BoolVariableInfo> Bools(ConstantBufferInfo &info) { return ArrayPairRef<BoolVariableInfo>(info.BoolCount, info.Bools.get()); }
ArrayPairRef<StructVariableInfo> Structs(ConstantBufferInfo &info) { return ArrayPairRef<StructVariableInfo>(info.StructCount, info.Structs.get()); }

ArrayPairRef<ConstantBufferInfo> ConstantBuffers(EffectResourceInfo &info) { return ArrayPairRef<ConstantBufferInfo>(info.ConstantBufferCount, info.ConstantBuffers.get()); }
ArrayPairRef<ConstantBufferInfo> TextureBuffers(EffectResourceInfo &info) { return ArrayPairRef<ConstantBufferInfo>(info.TextureBufferCount, info.TextureBuffers.get()); }
ArrayPairRef<ResourceInfo> Resources(EffectResourceInfo &info) { return ArrayPairRef<ResourceInfo>(info.ResourceCount, info.Resources.get()); }
ArrayPairRef<ResourceInfo> UAVs(EffectResourceInfo &info) { return ArrayPairRef<ResourceInfo>(info.UAVCount, info.UAVs.get()); }
ArrayPairRef<ResourceInfo> RenderTargets(EffectResourceInfo &info) { return ArrayPairRef<ResourceInfo>(info.RenderTargetCount, info.RenderTargets.get()); }
ArrayPairRef<ResourceInfo> DepthStencilTargets(EffectResourceInfo &info) { return ArrayPairRef<ResourceInfo>(info.DepthStencilTargetCount, info.DepthStencilTargets.get()); }

ArrayPairRef<ConstantBufferBindInfo> ConstantBuffers(ShaderBindingInfo &info) { return ArrayPairRef<ConstantBufferBindInfo>(info.ConstantBufferCount, info.ConstantBuffers.get()); }
ArrayPairRef<ConstantBufferBindInfo> TextureBuffers(ShaderBindingInfo &info) { return ArrayPairRef<ConstantBufferBindInfo>(info.TextureBufferCount, info.TextureBuffers.get()); }
ArrayPairRef<ResourceBindInfo> Resources(ShaderBindingInfo &info) { return ArrayPairRef<ResourceBindInfo>(info.ResourceCount, info.Resources.get()); }
ArrayPairRef<ResourceBindInfo> UAVs(ShaderBindingInfo &info) { return ArrayPairRef<ResourceBindInfo>(info.UAVCount, info.UAVs.get()); }
ArrayPairRef<SamplerStateBindInfo> Samplers(ShaderBindingInfo &info) { return ArrayPairRef<SamplerStateBindInfo>(info.SamplerCount, info.Samplers.get()); }

ArrayPairRef<ShaderInfo> VertexShaders(EffectShaderInfo &info) { return ArrayPairRef<ShaderInfo>(info.VertexShaderCount, info.VertexShaders.get()); }
ArrayPairRef<ShaderInfo> PixelShaders(EffectShaderInfo &info) { return ArrayPairRef<ShaderInfo>(info.PixelShaderCount, info.PixelShaders.get()); }
ArrayPairRef<ShaderInfo> GeometryShaders(EffectShaderInfo &info) { return ArrayPairRef<ShaderInfo>(info.GeometryShaderCount, info.GeometryShaders.get()); }
ArrayPairRef<ShaderInfo> ComputeShaders(EffectShaderInfo &info) { return ArrayPairRef<ShaderInfo>(info.ComputeShaderCount, info.ComputeShaders.get()); }
ArrayPairRef<ShaderInfo> HullShaders(EffectShaderInfo &info) { return ArrayPairRef<ShaderInfo>(info.HullShaderCount, info.HullShaders.get()); }
ArrayPairRef<ShaderInfo> DomainShaders(EffectShaderInfo &info) { return ArrayPairRef<ShaderInfo>(info.DomainShaderCount, info.DomainShaders.get()); }
ArrayPairRef<StreamOutInfo> StreamOut(EffectShaderInfo &info) { return ArrayPairRef<StreamOutInfo>(info.StreamOutCount, info.StreamOut.get()); }

ArrayPairRef<UINT> BufferStrides(StreamOutInfo &info) { return ArrayPairRef<UINT>(info.BufferCount, info.BufferStrides.get()); }
ArrayPairRef<D3D11_SO_DECLARATION_ENTRY> Elements(StreamOutInfo &info) { return ArrayPairRef<D3D11_SO_DECLARATION_ENTRY>(info.ElementCount, info.Elements.get()); }

ArrayPairRef<SamplerStateInfo> SamplerState(EffectStateInfo &info) { return ArrayPairRef<SamplerStateInfo>(info.SamplerStateCount, info.SamplerState.get()); }
ArrayPairRef<RasterizerStateInfo> RasterizerState(EffectStateInfo &info) { return ArrayPairRef<RasterizerStateInfo>(info.RasterizerStateCount, info.RasterizerState.get()); }
ArrayPairRef<DepthStencilStateInfo> DepthStencilState(EffectStateInfo &info) { return ArrayPairRef<DepthStencilStateInfo>(info.DepthStencilStateCount, info.DepthStencilState.get()); }
ArrayPairRef<BlendStateInfo> BlendState(EffectStateInfo &info) { return ArrayPairRef<BlendStateInfo>(info.BlendStateCount, info.BlendState.get()); }

// OPTIMIZE: Use variable index hint?
VariableInfo* FindVariable(EffectInfo &info, const char *name, UINT *pLiteralIndex = nullptr)
{
	if (info.Variables)
	{
		size_t n = strlen(name);

		if (n == 0)
			return nullptr;

		if (pLiteralIndex && name[n - 1] == ']')
		{
			while (n > 0 && name[n] != '[') --n;
			*pLiteralIndex = atoi(&name[n + 1]);
		}

		for (UINT i = 0; i < info.VariableCount; ++i)
			if (strncmp(info.Variables[i].Name, name, n) == 0)
				return &info.Variables[i];
	}

	return nullptr;
}

struct ShaderAux
{
	ShaderInfo *Shader;
	com_ptr<ID3D11ShaderReflection> Reflection;
};

struct ScratchPool
{
	AnnotationBlockInfo Annotations;
	ShaderBindingInfo ShaderBindings;
	ScratchArrayPair<ShaderAux> Shaders;
	StreamOutInfo StreamOut;
};

template <class InfoType>
UINT& BlockAnnotationCount(AnnotationBlockInfo *pBlockInfo);
template <>
UINT& BlockAnnotationCount<FloatAnnotationInfo>(AnnotationBlockInfo *pBlockInfo) { return pBlockInfo->FloatCount; }
template <>
UINT& BlockAnnotationCount<IntAnnotationInfo>(AnnotationBlockInfo *pBlockInfo) { return pBlockInfo->IntCount; }
template <>
UINT& BlockAnnotationCount<BoolAnnotationInfo>(AnnotationBlockInfo *pBlockInfo) { return pBlockInfo->BoolCount; }

template <class DataType, class InfoType>
InfoType* LoadAnnotation(Reader &unstructuredReader, Heap &heap,
						 D3DEffectsLite::AnnotationBlockInfo *pBlockInfo, 
						 const char *name, const D3DX11Effects::SBinaryType &type, const D3DX11Effects::SBinaryNumericType &numType, UINT dataOffset,
						 ArrayPairRef<InfoType> annotations)
{
	InfoType *pVariableInfo = AddAndMaybeConstruct(annotations);

	UINT valueCount = max(type.Elements, 1U) * numType.Rows * numType.Columns;
	assert(valueCount * sizeof(DataType) == type.PackedSize);
	DataType *values = MoveMultiple<DataType>(heap, FetchMultiple<DataType>(unstructuredReader, dataOffset, valueCount), valueCount);

	if (pVariableInfo)
	{
		pVariableInfo->Name = name;
		pVariableInfo->ValueCount = valueCount;
		pVariableInfo->Values = values;
	}

	return pVariableInfo;
}

D3DEffectsLite::AnnotationBlockInfo* LoadAnnotations(Reader &reader, Reader &unstructuredReader, Heap &heap, AnnotationBlockInfo &annotationPool)
{
	UINT annotationCount = Read<UINT>(reader);

	// Optional
	if (annotationCount == 0)
		return nullptr;

	D3DEffectsLite::AnnotationBlockInfo *pBlockInfo = heap.NoConstruct<AnnotationBlockInfo>();

	D3DEffectsLite::FloatAnnotationInfo *pFloatsBase = BasePointer(Floats(annotationPool));
	D3DEffectsLite::IntAnnotationInfo *pIntsBase = BasePointer(Ints(annotationPool));
	D3DEffectsLite::BoolAnnotationInfo *pBoolsBase = BasePointer(Bools(annotationPool));
	D3DEffectsLite::StringAnnotationInfo *pStringsBase = BasePointer(Strings(annotationPool));

	// Load annotations
	for (UINT annotationIdx = 0; annotationIdx < annotationCount; ++annotationIdx)
	{
		const D3DX11Effects::SBinaryAnnotation &annot = Read<D3DX11Effects::SBinaryAnnotation>(reader);
		char *name = MoveString(heap, FetchMultiple<char>(unstructuredReader, annot.oName, 1));
		
		unstructuredReader.pointer = annot.oType;
		const D3DX11Effects::SBinaryType &type = Read<D3DX11Effects::SBinaryType>(unstructuredReader);
		
		bool bSupportedType = false;

		if (type.VarType == D3DX11Effects::EVT_Numeric)
		{
			const D3DX11Effects::SBinaryNumericType &numType = Read<D3DX11Effects::SBinaryNumericType>(unstructuredReader);

			bSupportedType = true;
			
			switch (numType.ScalarType)
			{
			case D3DX11Effects::EST_Float:
				LoadAnnotation<FLOAT>(unstructuredReader, heap, pBlockInfo, name, type, numType, Read<UINT>(reader), Floats(annotationPool));
				break;
			case D3DX11Effects::EST_Int:
			case D3DX11Effects::EST_UInt:
				LoadAnnotation<INT>(unstructuredReader, heap, pBlockInfo, name, type, numType, Read<UINT>(reader), Ints(annotationPool));
				break;
			case D3DX11Effects::EST_Bool:
				LoadAnnotation<BOOL>(unstructuredReader, heap, pBlockInfo, name, type, numType, Read<UINT>(reader), Bools(annotationPool));
				break;
			default:
				bSupportedType = false;
			}
		}
		else if (type.VarType == D3DX11Effects::EVT_Object)
		{
			D3DX11Effects::EObjectType objType = Read<D3DX11Effects::EObjectType>(unstructuredReader);

			if (objType == D3DX11Effects::EOT_String)
			{
				bSupportedType = true;

				StringAnnotationInfo *pVariableInfo = AddAndMaybeConstruct(Strings(annotationPool));
				
				UINT stringCount = max(type.Elements, 1U);
				const char **data = heap.NoConstructMultiple<const char*>(stringCount);

				for (UINT i = 0; i < stringCount; ++i)
				{
					const char *string = MoveString(heap, FetchMultiple<char>(unstructuredReader, Read<UINT>(reader), 1));
					if (data) data[i] = string;
				}

				if (pVariableInfo)
				{
					pVariableInfo->Name = name;
					pVariableInfo->ValueCount = stringCount;
					pVariableInfo->Values = data;
				}
			}
		}
		else if (type.VarType == D3DX11Effects::EVT_Struct)
		{
			bSupportedType = true;

			// TODO: Official FX supports this, implement?
			D3DEFFECTSLITE_LOG_LINE("Structure annotations unsupported, ignored");

			// TODO: Skip data offset
			Read<UINT>(reader);
		}

		Check(bSupportedType, D3DEFFECTSLITE_MAKE_LINE("Unsupported annotation type"));
	}

	if (pBlockInfo)
	{
		Floats(*pBlockInfo) = Range(Floats(annotationPool), pFloatsBase);
		Ints(*pBlockInfo) = Range(Ints(annotationPool), pIntsBase);
		Bools(*pBlockInfo)= Range(Bools(annotationPool), pBoolsBase);
		Strings(*pBlockInfo) = Range(Strings(annotationPool), pStringsBase);
	}

	return pBlockInfo;
}

template <class InfoType>
UINT& ConstantCount(ConstantBufferInfo *pCBInfo);
template <>
UINT& ConstantCount<FloatVariableInfo>(ConstantBufferInfo *pCBInfo) { return pCBInfo->FloatCount; }
template <>
UINT& ConstantCount<IntVariableInfo>(ConstantBufferInfo *pCBInfo) { return pCBInfo->IntCount; }
template <>
UINT& ConstantCount<BoolVariableInfo>(ConstantBufferInfo *pCBInfo) { return pCBInfo->BoolCount; }

template <class DataType, class InfoType>
InfoType* LoadConstant(Reader &unstructuredReader, Heap &heap,
					   ConstantBufferInfo *pCBInfo,
					   VariableInfo *pVariableInfo, const D3DX11Effects::SBinaryNumericVariable &variable,
					   const D3DX11Effects::SBinaryType &type, const D3DX11Effects::SBinaryNumericType &numType,
					   ArrayPairRef<InfoType> constants)
{
	InfoType *pConstantInfo = AddAndMaybeConstruct(constants);

	UINT valueCount = max(type.Elements, 1U) * numType.Rows * numType.Columns;
	assert(valueCount * sizeof(DataType) == type.PackedSize);
	DataType *pValues = (variable.oDefaultValue != 0)
		? MoveMultiple<DataType>(heap, FetchMultiple<DataType>(unstructuredReader, variable.oDefaultValue, valueCount), valueCount)
		: nullptr;

	if (pConstantInfo)
	{
		pConstantInfo->Variable = pVariableInfo;

		pConstantInfo->Constant.Offset = variable.Offset;
		pConstantInfo->Constant.Stride = type.Stride;
		pConstantInfo->Constant.Rows = static_cast<USHORT>(numType.Rows);
		pConstantInfo->Constant.Columns = static_cast<USHORT>(numType.Columns);
		pConstantInfo->Constant.Elements = max(type.Elements, 1U);

		pConstantInfo->ValueCount = valueCount;
		pConstantInfo->Values = pValues;
	}

	return pConstantInfo;
}

void LoadNumericVariables(Reader &reader, Reader &unstructuredReader, const D3DX11Effects::SBinaryHeader &header, PlainReflection &reflection, ScratchPool &scratchPool,
						  UINT variableCount, ConstantBufferInfo *pCBInfo)
{
	EffectInfo &info = reflection.Info;
	Heap &heap = reflection.Heap;

	D3DEffectsLite::FloatVariableInfo *pFloatsBase = BasePointer(Floats(info.Constants));
	D3DEffectsLite::IntVariableInfo *pIntsBase = BasePointer(Ints(info.Constants));
	D3DEffectsLite::BoolVariableInfo *pBoolsBase = BasePointer(Bools(info.Constants));
	D3DEffectsLite::StructVariableInfo *pStructsBase = BasePointer(Structs(info.Constants));

	// Load variables
	for (UINT variableIdx = 0; variableIdx < variableCount; ++variableIdx)
	{
		const D3DX11Effects::SBinaryNumericVariable &var = Read<D3DX11Effects::SBinaryNumericVariable>(reader);
		VariableInfo *pVariableInfo = AddAndMaybeConstruct(Variables(info));
		char *name = MoveString(heap, FetchMultiple<char>(unstructuredReader, var.oName, 1));
		char *pSemantic = (var.oSemantic != 0) ? MoveString(heap, FetchMultiple<char>(unstructuredReader, var.oSemantic, 1)) : nullptr;
		
		D3D_SHADER_VARIABLE_TYPE variableType;
		void *untypedConstantInfo;

		unstructuredReader.pointer = var.oType;
		const D3DX11Effects::SBinaryType &type = Read<D3DX11Effects::SBinaryType>(unstructuredReader);
		
		// WARNING: SVT_VOID is alias of SVT_STRUCT, therefore need separate flag
		bool bSupportedType = false;

		if (type.VarType == D3DX11Effects::EVT_Numeric)
		{
			const D3DX11Effects::SBinaryNumericType &numType = Read<D3DX11Effects::SBinaryNumericType>(unstructuredReader);

			bSupportedType = true;

			switch (numType.ScalarType)
			{
			case D3DX11Effects::EST_Float:
				variableType = D3D_SVT_FLOAT;
				untypedConstantInfo = LoadConstant<FLOAT>(unstructuredReader, heap, pCBInfo,
					pVariableInfo, var, type, numType, Floats(info.Constants));
				break;
			case D3DX11Effects::EST_Int:
			case D3DX11Effects::EST_UInt:
				variableType = D3D_SVT_INT;
				untypedConstantInfo = LoadConstant<INT>(unstructuredReader, heap, pCBInfo,
					pVariableInfo, var, type, numType, Ints(info.Constants));
				break;
			case D3DX11Effects::EST_Bool:
				variableType = D3D_SVT_BOOL;
				untypedConstantInfo = LoadConstant<BOOL>(unstructuredReader, heap, pCBInfo,
					pVariableInfo, var, type, numType, Bools(info.Constants));
				break;
			default:
				bSupportedType = false;
			}
		}
		else if (type.VarType == D3DX11Effects::EVT_Struct)
		{
			bSupportedType = true;

			StructVariableInfo *pConstantInfo = AddAndMaybeConstruct(Structs(info.Constants));
			variableType = D3D_SVT_STRUCT;
			untypedConstantInfo = pConstantInfo;

			void *pData = (var.oDefaultValue != 0)
				? MoveData(heap, FetchMultiple<BYTE>(unstructuredReader, var.oDefaultValue, type.PackedSize), type.PackedSize)
				: nullptr;

			if (pConstantInfo)
			{
				pConstantInfo->Variable = pVariableInfo;

				pConstantInfo->Constant.Offset = var.Offset;
				pConstantInfo->Constant.Stride = type.Stride;
				pConstantInfo->Constant.Rows = 1;
				pConstantInfo->Constant.Columns = 1;
				pConstantInfo->Constant.Elements = max(type.Elements, 1U);

				pConstantInfo->ByteCount = type.PackedSize;
				pConstantInfo->Bytes = pData;
			}
		}

		Check(bSupportedType, D3DEFFECTSLITE_MAKE_LINE("Unsupported constant type"));

		AnnotationBlockInfo *pAnnotations = LoadAnnotations(reader, unstructuredReader, heap, scratchPool.Annotations);

		if (pVariableInfo)
		{
			pVariableInfo->Name = name;
			pVariableInfo->Semantic = pSemantic;
			pVariableInfo->Annotations = pAnnotations;
			pVariableInfo->Type = variableType;
			pVariableInfo->Info = untypedConstantInfo;
		}
	}

	if (pCBInfo)
	{
		Floats(*pCBInfo) = Range(Floats(info.Constants), pFloatsBase);
		Ints(*pCBInfo) = Range(Ints(info.Constants), pIntsBase);
		Bools(*pCBInfo)= Range(Bools(info.Constants), pBoolsBase);
		Structs(*pCBInfo) = Range(Structs(info.Constants), pStructsBase);
	}
}

void LoadConstantBuffers(Reader &reader, Reader &unstructuredReader, const D3DX11Effects::SBinaryHeader &header, PlainReflection &reflection, ScratchPool &scratchPool)
{
	EffectInfo &info = reflection.Info;
	Heap &heap = reflection.Heap;

	for (UINT constantBufferIdx = 0; constantBufferIdx < header.Effect.cCBs; ++constantBufferIdx)
	{
		const D3DX11Effects::SBinaryConstantBuffer &cb = Read<D3DX11Effects::SBinaryConstantBuffer>(reader);
		VariableInfo *pVariableInfo = AddAndMaybeConstruct(Variables(info));
		char *name = MoveString(heap, FetchMultiple<char>(unstructuredReader, cb.oName, 1));
		
		bool isTextureBuffer = ((cb.Flags & cb.c_IsTBuffer) != 0);
		ConstantBufferInfo *pCBInfo = AddAndMaybeConstruct( (isTextureBuffer) ? TextureBuffers(info.Resources) : ConstantBuffers(info.Resources) );

		if (pCBInfo)
		{
			pCBInfo->BindPoint = cb.ExplicitBindPoint;
			pCBInfo->Size = AlignInteger(cb.Size, ConstantBufferAlignment);
			pCBInfo->Variable = pVariableInfo;
		}

		AnnotationBlockInfo *pAnnotations = LoadAnnotations(reader, unstructuredReader, heap, scratchPool.Annotations);

		if (pVariableInfo)
		{
			pVariableInfo->Name = name;
			pVariableInfo->Semantic = nullptr;
			pVariableInfo->Annotations = pAnnotations;
			pVariableInfo->Type = isTextureBuffer ? D3D_SVT_TBUFFER : D3D_SVT_CBUFFER;
			pVariableInfo->Info = pCBInfo;
		}

		LoadNumericVariables(reader, unstructuredReader, header, reflection, scratchPool, cb.cVariables, pCBInfo);
	}
}

StringVariableInfo* LoadStringConstant(Reader &reader, Reader &unstructuredReader, PlainReflection &reflection,
									   VariableInfo *pVariableInfo, const D3DX11Effects::SBinaryType &type)
{
	EffectInfo &info = reflection.Info;
	Heap &heap = reflection.Heap;

	StringVariableInfo *pConstantInfo = AddAndMaybeConstruct(Strings(info.Constants));

	UINT stringCount = max(type.Elements, 1U);
	const char **data = heap.NoConstructMultiple<const char*>(stringCount);

	for (UINT i = 0; i < stringCount; ++i)
	{
		const char *string = MoveString(heap, FetchMultiple<char>(unstructuredReader, Read<UINT>(reader), 1));
		if (data) data[i] = string;
	}

	if (pConstantInfo)
	{
		pConstantInfo->Variable = pVariableInfo;
		pConstantInfo->ValueCount = stringCount;
		pConstantInfo->Values = data;
	}

	return pConstantInfo;
}

D3D_SHADER_VARIABLE_TYPE ToSRVType(D3DX11Effects::EObjectType objType)
{
	switch (objType)
	{
	case D3DX11Effects::EOT_Texture: return D3D_SVT_TEXTURE;
	case D3DX11Effects::EOT_Texture1D: return D3D_SVT_TEXTURE1D;
	case D3DX11Effects::EOT_Texture1DArray: return D3D_SVT_TEXTURE1DARRAY;
	case D3DX11Effects::EOT_Texture2D: return D3D_SVT_TEXTURE2D;
	case D3DX11Effects::EOT_Texture2DArray: return D3D_SVT_TEXTURE2DARRAY;
	case D3DX11Effects::EOT_Texture2DMS: return D3D_SVT_TEXTURE2DMS;
	case D3DX11Effects::EOT_Texture2DMSArray: return D3D_SVT_TEXTURE2DMSARRAY;
	case D3DX11Effects::EOT_Texture3D: return D3D_SVT_TEXTURE3D;
	case D3DX11Effects::EOT_TextureCube: return D3D_SVT_TEXTURECUBE;
	case D3DX11Effects::EOT_TextureCubeArray: return D3D_SVT_TEXTURECUBEARRAY;
	case D3DX11Effects::EOT_Buffer: return D3D_SVT_BUFFER;
	case D3DX11Effects::EOT_StructuredBuffer: return D3D_SVT_STRUCTURED_BUFFER;
	case D3DX11Effects::EOT_ByteAddressBuffer: return D3D_SVT_BYTEADDRESS_BUFFER;
	default: return D3D_SVT_VOID;
	}
}

D3D_SHADER_VARIABLE_TYPE ToUAVType(D3DX11Effects::EObjectType objType)
{
	switch (objType)
	{
	case D3DX11Effects::EOT_RWTexture1D: return D3D_SVT_RWTEXTURE1D;
	case D3DX11Effects::EOT_RWTexture1DArray: return D3D_SVT_RWTEXTURE1DARRAY;
	case D3DX11Effects::EOT_RWTexture2D: return D3D_SVT_RWTEXTURE2D;
	case D3DX11Effects::EOT_RWTexture2DArray: return D3D_SVT_RWTEXTURE2DARRAY;
	case D3DX11Effects::EOT_RWTexture3D: return D3D_SVT_RWTEXTURE3D;
	case D3DX11Effects::EOT_RWBuffer: return D3D_SVT_RWBUFFER;
	case D3DX11Effects::EOT_RWStructuredBuffer: return D3D_SVT_RWSTRUCTURED_BUFFER;
	case D3DX11Effects::EOT_RWStructuredBufferAlloc: return D3D_SVT_RWSTRUCTURED_BUFFER;
	case D3DX11Effects::EOT_RWStructuredBufferConsume: return D3D_SVT_RWSTRUCTURED_BUFFER;
	case D3DX11Effects::EOT_AppendStructuredBuffer: return D3D_SVT_APPEND_STRUCTURED_BUFFER;
	case D3DX11Effects::EOT_ConsumeStructuredBuffer: return D3D_SVT_CONSUME_STRUCTURED_BUFFER;
	case D3DX11Effects::EOT_RWByteAddressBuffer: return D3D_SVT_RWBYTEADDRESS_BUFFER;
	default: return D3D_SVT_VOID;
	}
}

D3D_SHADER_VARIABLE_TYPE ToRenderTargetType(D3DX11Effects::EObjectType objType)
{
	switch (objType)
	{
	case D3DX11Effects::EOT_RenderTargetView: return D3D_SVT_RENDERTARGETVIEW;
	default: return D3D_SVT_VOID;
	}
}

D3D_SHADER_VARIABLE_TYPE ToDepthStencilTargetType(D3DX11Effects::EObjectType objType)
{
	switch (objType)
	{
	case D3DX11Effects::EOT_DepthStencilView: return D3D_SVT_DEPTHSTENCILVIEW;
	default: return D3D_SVT_VOID;
	}
}

ResourceInfo* LoadResource(PlainReflection &reflection,
						   VariableInfo *pVariableInfo, const D3DX11Effects::SBinaryObjectVariable &variable, const D3DX11Effects::SBinaryType &type, 
						   ArrayPairRef<ResourceInfo> resources)
{
	Heap &heap = reflection.Heap;

	ResourceInfo *pFirstResourceInfo = nullptr;
	UINT successorCount = max(type.Elements, 1U);

	while (successorCount-- > 0)
	{
		ResourceInfo *pResourceInfo = AddAndMaybeConstruct(resources);
		if (!pFirstResourceInfo) pFirstResourceInfo = pResourceInfo;

		if (pResourceInfo)
		{
			pResourceInfo->Variable = pVariableInfo;
			pResourceInfo->BindPoint = variable.ExplicitBindPoint;
			pResourceInfo->SuccessorCount = successorCount;
		}
	}

	return pFirstResourceInfo;
}

D3D_SHADER_VARIABLE_TYPE ToShaderType(D3DX11Effects::EObjectType objType)
{
	switch (objType)
	{
	case D3DX11Effects::EOT_VertexShader:
	case D3DX11Effects::EOT_VertexShader5: return D3D_SVT_VERTEXSHADER;
	case D3DX11Effects::EOT_PixelShader: 
	case D3DX11Effects::EOT_PixelShader5: return D3D_SVT_PIXELSHADER;
	case D3DX11Effects::EOT_GeometryShader: 
	case D3DX11Effects::EOT_GeometryShaderSO: 
	case D3DX11Effects::EOT_GeometryShader5: return D3D_SVT_GEOMETRYSHADER;
	case D3DX11Effects::EOT_ComputeShader5: return D3D_SVT_COMPUTESHADER;
	case D3DX11Effects::EOT_HullShader5: return D3D_SVT_HULLSHADER;
	case D3DX11Effects::EOT_DomainShader5: return D3D_SVT_DOMAINSHADER;
	default: return D3D_SVT_VOID;
	}
}

ArrayPairRef<ShaderInfo> Shaders(EffectShaderInfo &info, D3D_SHADER_VARIABLE_TYPE svt)
{
	switch (svt)
	{
	case D3D_SVT_VERTEXSHADER: return VertexShaders(info);
	case D3D_SVT_PIXELSHADER: return PixelShaders(info);
	case D3D_SVT_GEOMETRYSHADER: return GeometryShaders(info);
	case D3D_SVT_COMPUTESHADER: return ComputeShaders(info);
	case D3D_SVT_HULLSHADER:  return HullShaders(info);
	case D3D_SVT_DOMAINSHADER: return DomainShaders(info);
	default: assert(false); D3DEFFECTSLITE_ASSUME(false);
	}
}

bool IsShader4(D3DX11Effects::EObjectType objType)
{
	switch (objType)
	{
	case D3DX11Effects::EOT_VertexShader:
	case D3DX11Effects::EOT_PixelShader:
	case D3DX11Effects::EOT_GeometryShader:
	case D3DX11Effects::EOT_GeometryShaderSO: return true;
	default: return false;
	}
}

StreamOutInfo* LoadStreamOut(Reader &unstructuredReader, PlainReflection &reflection, ScratchPool &scratchPool,
							 ShaderInfo *pShaderInfo, UINT declCount, const UINT *declOffsets, UINT rasterizedStream)
{
	Heap &heap = reflection.Heap;

	D3D11_SO_DECLARATION_ENTRY *pElementsBase = BasePointer(Elements(scratchPool.StreamOut));

	StreamOutInfo *pStreamOutInfo = AddAndMaybeConstruct(StreamOut(reflection.Info.Shaders));

	UINT bufferCount = 0;
	UINT bufferStrideBuffer[D3D11_SO_BUFFER_SLOT_COUNT] = { 0 };

	for (UINT i = 0; i < declCount; ++i)
	{
		const char *SODecl = FetchMultiple<char>(unstructuredReader, declOffsets[i], 1);
		size_t SODeclLen = strlen(SODecl);

		UINT elementCount = static_cast<UINT>( strcnt(SODecl, ';') + 1 );
		D3D11_SO_DECLARATION_ENTRY *pElements = AddAndMaybeConstructMultiple(Elements(scratchPool.StreamOut), elementCount);

		const char *SODeclPtr = SODecl;

		for (UINT j = 0; j < elementCount; ++j)
		{
			D3D11_SO_DECLARATION_ENTRY element;
			memset(&element, 0, sizeof(element));
			element.Stream = i;
			element.ComponentCount = 4;

			const char *semanticBegin = strna(SODeclPtr);
			const char *semanticEnd = strnna(semanticBegin);
			const char *elementEnd = strnc(SODeclPtr, ';');
			Check(semanticBegin < semanticEnd && semanticEnd <= elementEnd, D3DEFFECTSLITE_MAKE_LINE("Expected stream out semantic name"));

			// Output slot
			// Always: Counts buffers!
			{
				const char *colon = strnc(SODeclPtr, ':');
				
				if (colon < semanticBegin)
				{
					const char *outSlotBegin = strnd(SODeclPtr);
					Check(outSlotBegin < colon, D3DEFFECTSLITE_MAKE_LINE("Expected stream out buffer slot"));
					element.OutputSlot = atoi(outSlotBegin);
					Check(element.OutputSlot < arraylen(bufferStrideBuffer), D3DEFFECTSLITE_MAKE_LINE("Stream out buffer slot out of range"));
				}

				bufferCount = max(bufferCount, (UINT) element.OutputSlot + 1);
			}

			// Semantic
			// ALWAYS: Moves data!
			{
				size_t nameLength = semanticEnd - semanticBegin;

				// Allow for skipping of buffer space
				if (semanticBegin == SODeclPtr || strncmp(semanticBegin - 1, "$SKIP", nameLength + 1) != 0)
				{
					char *pName = MoveMultiple(heap, semanticBegin, nameLength + 1);
					if (pName)
					{
						pName[nameLength] = 0;
						element.SemanticName = pName;
					}
				}
			}
			
			// Semantic Index
			if (pElements)
			{
				const char *indexBegin = strnd(semanticEnd);

				if (indexBegin < elementEnd)
					element.SemanticIndex = atoi(indexBegin);
			}
			
			// Components
			if (pElements)
			{
				const char *dot = strnc(semanticEnd, '.');

				if (dot < elementEnd)
				{
					const char *mask = strna(dot);

					element.StartComponent = 4;
					element.ComponentCount = 0;

					while (mask < elementEnd)
						switch (*mask++)
						{
						case 'x': element.StartComponent = min(element.StartComponent, 0); element.ComponentCount = max(element.ComponentCount, 1); break;
						case 'y': element.StartComponent = min(element.StartComponent, 1); element.ComponentCount = max(element.ComponentCount, 2); break;
						case 'z': element.StartComponent = min(element.StartComponent, 2); element.ComponentCount = max(element.ComponentCount, 3); break;
						case 'w': element.StartComponent = min(element.StartComponent, 3); element.ComponentCount = max(element.ComponentCount, 4); break;
						}

					Check(element.StartComponent < element.ComponentCount, D3DEFFECTSLITE_MAKE_LINE("Expected stream out write mask"));
					element.ComponentCount -= element.StartComponent;
				}

				bufferStrideBuffer[element.OutputSlot] += sizeof(FLOAT) * element.ComponentCount;
			}

			SODeclPtr = (*elementEnd) ? elementEnd + 1 : elementEnd;

			if (pElements)
				pElements[j] = element;
		}
	}

	UINT *bufferStrides = MoveMultiple<UINT>(heap, bufferStrideBuffer, bufferCount);

	if (pStreamOutInfo)
	{
		pStreamOutInfo->Shader = pShaderInfo;
		pStreamOutInfo->RasterizedStream = rasterizedStream;

		Elements(*pStreamOutInfo) = Range(Elements(scratchPool.StreamOut), pElementsBase);
		BufferStrides(*pStreamOutInfo) = ArrayPair<UINT>(bufferCount, bufferStrides);
	}

	return pStreamOutInfo;
}

ShaderInfo* LoadShader(Reader &reader, Reader &unstructuredReader, PlainReflection &reflection, ScratchPool &scratchPool,
					   VariableInfo *pVariableInfo, const D3DX11Effects::SBinaryType &type, D3DX11Effects::EObjectType objType,
					   ArrayPairRef<ShaderInfo> shaders)
{
	Heap &heap = reflection.Heap;

	ShaderInfo *pFirstShaderInfo = nullptr;
	UINT successorCount = max(type.Elements, 1U);

	while (successorCount-- > 0)
	{
		ShaderInfo *pShaderInfo = AddAndMaybeConstruct(shaders);
		if (!pFirstShaderInfo) pFirstShaderInfo = pShaderInfo;

		StreamOutInfo *pStreamOutInfo = nullptr;

		UINT shaderOffset;

		if (IsShader4(objType))
		{
			if (objType == D3DX11Effects::EOT_GeometryShaderSO)
			{
				const D3DX11Effects::SBinaryGSSOInitializer &geometryShaderSO4 = Read<D3DX11Effects::SBinaryGSSOInitializer>(reader);
				shaderOffset = geometryShaderSO4.oShader;

				pStreamOutInfo = LoadStreamOut(unstructuredReader, reflection, scratchPool, pShaderInfo, 1, &geometryShaderSO4.oSODecl, 0);
			}
			else
				shaderOffset = Read<UINT>(reader);
		}
		else
		{
			const D3DX11Effects::SBinaryShaderData5 &shader5 = Read<D3DX11Effects::SBinaryShaderData5>(reader);
			shaderOffset = shader5.oShader;

			pStreamOutInfo = LoadStreamOut(unstructuredReader, reflection, scratchPool, pShaderInfo, shader5.cSODecls, shader5.oSODecls, shader5.RasterizedStream);
		}

		unstructuredReader.pointer = shaderOffset;
		UINT byteCount = Read<UINT>(unstructuredReader);
		const void *inlineByteCode = ReadMultiple<BYTE>(unstructuredReader, byteCount);
		void *byteCode = MoveData(heap, inlineByteCode, byteCount);

		UINT shaderIdx = scratchPool.Shaders.Count++;
		scratchPool.Shaders.Array[shaderIdx].Shader = pShaderInfo;

		// Create reflection, if not yet existent
		if (scratchPool.Shaders.Array[shaderIdx].Reflection.get() == nullptr)
		{
			HRESULT reflectionResult = ::D3DReflect(inlineByteCode, byteCount, IID_ID3D11ShaderReflection, (void**)&scratchPool.Shaders.Array[shaderIdx].Reflection);
			Check(reflectionResult == S_OK, D3DEFFECTSLITE_MAKE_LINE("Cannot create shader reflection object"));
		}

		if (pShaderInfo)
		{
			pShaderInfo->Variable = pVariableInfo;
			pShaderInfo->ByteCount = byteCount;
			pShaderInfo->ByteCode = byteCode;
			pShaderInfo->StreamOut = pStreamOutInfo;
			pShaderInfo->SuccessorCount = successorCount;
		}
	}

	return pFirstShaderInfo;
}

D3D_SHADER_VARIABLE_TYPE ToStateType(D3DX11Effects::EObjectType objType)
{
	switch (objType)
	{
	case D3DX11Effects::EOT_Sampler: return D3D_SVT_SAMPLER;
	case D3DX11Effects::EOT_Rasterizer: return D3D_SVT_RASTERIZER;
	case D3DX11Effects::EOT_DepthStencil: return D3D_SVT_DEPTHSTENCIL;
	case D3DX11Effects::EOT_Blend: return D3D_SVT_BLEND;
	default: return D3D_SVT_VOID;
	}
}

struct Assignment
{
	D3D_SHADER_VARIABLE_TYPE Type;
	UINT Elements;
	UINT IndexCount;
	UINT DataOffset;
	UINT DataStride;
};

const UINT RasterizerAssignmentsBaseIndex = 12;
const Assignment RasterizerAssignments[] =
{
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(RasterizerStateInfo, Desc.FillMode), 0 },
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(RasterizerStateInfo, Desc.CullMode), 0 },
	{ D3D_SVT_BOOL, 1, 1, offsetof_uint(RasterizerStateInfo, Desc.FrontCounterClockwise), 0 },
	{ D3D_SVT_INT, 1, 1, offsetof_uint(RasterizerStateInfo, Desc.DepthBias), 0 },
	{ D3D_SVT_FLOAT, 1, 1, offsetof_uint(RasterizerStateInfo, Desc.DepthBiasClamp), 0 },
	{ D3D_SVT_FLOAT, 1, 1, offsetof_uint(RasterizerStateInfo, Desc.SlopeScaledDepthBias), 0 },
	{ D3D_SVT_BOOL, 1, 1, offsetof_uint(RasterizerStateInfo, Desc.DepthClipEnable), 0 },
	{ D3D_SVT_BOOL, 1, 1, offsetof_uint(RasterizerStateInfo, Desc.ScissorEnable), 0 },
	{ D3D_SVT_BOOL, 1, 1, offsetof_uint(RasterizerStateInfo, Desc.MultisampleEnable), 0 },
	{ D3D_SVT_BOOL, 1, 1, offsetof_uint(RasterizerStateInfo, Desc.AntialiasedLineEnable), 0 }
};
const Assignment& RasterizerAssignment(UINT index)
{
	Check(RasterizerAssignmentsBaseIndex <= index && index - RasterizerAssignmentsBaseIndex < arraylen(RasterizerAssignments),
		D3DEFFECTSLITE_MAKE_LINE("Unsupported rasterizer state attribute"));
	return RasterizerAssignments[index - RasterizerAssignmentsBaseIndex];
}

const UINT DepthStencilAssignmentsBaseIndex = RasterizerAssignmentsBaseIndex + arraylen(RasterizerAssignments);
const Assignment DepthStencilAssignments[] =
{
	{ D3D10_SVT_BOOL, 1, 1, offsetof_uint(DepthStencilStateInfo, Desc.DepthEnable), 0 },
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(DepthStencilStateInfo, Desc.DepthWriteMask), 0 },
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(DepthStencilStateInfo, Desc.DepthFunc), 0 },
	{ D3D_SVT_BOOL, 1, 1, offsetof_uint(DepthStencilStateInfo, Desc.StencilEnable), 0 },
	{ D3D_SVT_UINT8, 1, 1, offsetof_uint(DepthStencilStateInfo, Desc.StencilReadMask), 0 },
	{ D3D_SVT_UINT8, 1, 1, offsetof_uint(DepthStencilStateInfo, Desc.StencilWriteMask), 0 },
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(DepthStencilStateInfo, Desc.FrontFace.StencilFailOp), 0 },
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(DepthStencilStateInfo, Desc.FrontFace.StencilDepthFailOp), 0 },
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(DepthStencilStateInfo, Desc.FrontFace.StencilPassOp), 0 },
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(DepthStencilStateInfo, Desc.FrontFace.StencilFunc), 0 },
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(DepthStencilStateInfo, Desc.BackFace.StencilFailOp), 0 },
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(DepthStencilStateInfo, Desc.BackFace.StencilDepthFailOp), 0 },
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(DepthStencilStateInfo, Desc.BackFace.StencilPassOp), 0 },
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(DepthStencilStateInfo, Desc.BackFace.StencilFunc), 0 }
};
const Assignment& DepthStencilAssignment(UINT index)
{
	Check(DepthStencilAssignmentsBaseIndex <= index && index - DepthStencilAssignmentsBaseIndex < arraylen(DepthStencilAssignments),
		D3DEFFECTSLITE_MAKE_LINE("Unsupported depth-stencil state attribute"));
	return DepthStencilAssignments[index - DepthStencilAssignmentsBaseIndex];
}

const UINT BlendAssignmentsBaseIndex = DepthStencilAssignmentsBaseIndex + arraylen(DepthStencilAssignments);
const Assignment BlendAssignments[] =
{
	{ D3D_SVT_BOOL, 1, 1, offsetof_uint(BlendStateInfo, Desc.AlphaToCoverageEnable), 0 },
	{ D3D_SVT_BOOL, 1, 8, offsetof_uint(BlendStateInfo, Desc.RenderTarget[0].BlendEnable), strideof_uint(BlendStateInfo, Desc.RenderTarget) },
	{ D3D_SVT_UINT, 1, 8, offsetof_uint(BlendStateInfo, Desc.RenderTarget[0].SrcBlend), strideof_uint(BlendStateInfo, Desc.RenderTarget) },
	{ D3D_SVT_UINT, 1, 8, offsetof_uint(BlendStateInfo, Desc.RenderTarget[0].DestBlend), strideof_uint(BlendStateInfo, Desc.RenderTarget) },
	{ D3D_SVT_UINT, 1, 8, offsetof_uint(BlendStateInfo, Desc.RenderTarget[0].BlendOp), strideof_uint(BlendStateInfo, Desc.RenderTarget) },
	{ D3D_SVT_UINT, 1, 8, offsetof_uint(BlendStateInfo, Desc.RenderTarget[0].SrcBlendAlpha), strideof_uint(BlendStateInfo, Desc.RenderTarget) },
	{ D3D_SVT_UINT, 1, 8, offsetof_uint(BlendStateInfo, Desc.RenderTarget[0].DestBlendAlpha), strideof_uint(BlendStateInfo, Desc.RenderTarget) },
	{ D3D_SVT_UINT, 1, 8, offsetof_uint(BlendStateInfo, Desc.RenderTarget[0].BlendOpAlpha), strideof_uint(BlendStateInfo, Desc.RenderTarget) },
	{ D3D_SVT_UINT8, 1, 8, offsetof_uint(BlendStateInfo, Desc.RenderTarget[0].RenderTargetWriteMask), strideof_uint(BlendStateInfo, Desc.RenderTarget) }
};
const Assignment& BlendAssignment(UINT index)
{
	Check(BlendAssignmentsBaseIndex <= index && index - BlendAssignmentsBaseIndex < arraylen(BlendAssignments),
		D3DEFFECTSLITE_MAKE_LINE("Unsupported blend state attribute"));
	return BlendAssignments[index - BlendAssignmentsBaseIndex];
}

const UINT SamplerAssignmentsBaseIndex = BlendAssignmentsBaseIndex + arraylen(BlendAssignments);
const Assignment SamplerAssignments[] =
{
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(SamplerStateInfo, Desc.Filter), 0 },
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(SamplerStateInfo, Desc.AddressU), 0 },
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(SamplerStateInfo, Desc.AddressV), 0 },
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(SamplerStateInfo, Desc.AddressW), 0 },
	{ D3D_SVT_FLOAT, 1, 1, offsetof_uint(SamplerStateInfo, Desc.MipLODBias), 0 },
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(SamplerStateInfo, Desc.MaxAnisotropy), 0 },
	{ D3D_SVT_UINT, 1, 1, offsetof_uint(SamplerStateInfo, Desc.ComparisonFunc), 0 },
	{ D3D_SVT_FLOAT, 4, 1, offsetof_uint(SamplerStateInfo, Desc.BorderColor), 0 },
	{ D3D_SVT_FLOAT, 1, 1, offsetof_uint(SamplerStateInfo, Desc.MinLOD), 0 },
	{ D3D_SVT_FLOAT, 1, 1, offsetof_uint(SamplerStateInfo, Desc.MaxLOD), 0 },
	// MONITOR: Old style texture assignment? Unsupported ...
	{ D3D_SVT_TEXTURE, 0, 0, 0, 0 }
};
const Assignment& SamplerAssignment(UINT index)
{
	Check(SamplerAssignmentsBaseIndex <= index && index - SamplerAssignmentsBaseIndex < arraylen(SamplerAssignments),
		D3DEFFECTSLITE_MAKE_LINE("Unsupported sampler state attribute"));
	index -= SamplerAssignmentsBaseIndex;
	Check(SamplerAssignments[index].Type != D3D_SVT_TEXTURE,
		D3DEFFECTSLITE_MAKE_LINE("Old-style texture assignment unsupported"));
	return SamplerAssignments[index];
}

typedef const Assignment& AssignmentSelector(UINT index);

void LoadStateAssignments(Reader &reader, Reader &unstructuredReader, void *pStateInfo, AssignmentSelector &stateAssignmentSelector)
{
	// Load assignments
	UINT assignmentCount = Read<UINT>(reader);
	const D3DX11Effects::SBinaryAssignment *assignments = ReadMultiple<D3DX11Effects::SBinaryAssignment>(reader, assignmentCount);

	// NOTE: Rest of assignments in unstructured memory + no allocations, only modifications
	if (!pStateInfo)
		return;

	for (UINT assignmentIdx = 0; assignmentIdx < assignmentCount; ++assignmentIdx)
	{
		const D3DX11Effects::SBinaryAssignment &assignment = assignments[assignmentIdx];
		const Assignment &stateAssignment = stateAssignmentSelector(assignment.iState);
		
		Check(assignment.Index < stateAssignment.IndexCount, D3DEFFECTSLITE_MAKE_LINE("State attribute index out of bounds"));
		BYTE *data = static_cast<BYTE*>(pStateInfo) + stateAssignment.DataOffset + (assignment.Index * stateAssignment.DataStride);

		if (assignment.AssignmentType == D3DX11Effects::ECAT_Constant)
		{
			unstructuredReader.pointer = assignment.oInitializer;
			UINT constantCount = Read<UINT>(unstructuredReader);
			const D3DX11Effects::SBinaryConstant* constants = ReadMultiple<D3DX11Effects::SBinaryConstant>(unstructuredReader, constantCount);

			Check(constantCount == stateAssignment.Elements, D3DEFFECTSLITE_MAKE_LINE("State attribute element count mismatch"));

			for (UINT constantIdx = 0; constantIdx < constantCount; ++constantIdx)
			{
				const D3DX11Effects::SBinaryConstant &constant = constants[constantIdx];

				switch (constant.Type)
				{
				case D3DX11Effects::EST_Float:
					Check(D3D_SVT_FLOAT == stateAssignment.Type, D3DEFFECTSLITE_MAKE_LINE("State attribute type mismatch, expected float"));
					reinterpret_cast<FLOAT*>(data)[constantIdx] = constant.fValue;
					break;
				case D3DX11Effects::EST_Int:
				case D3DX11Effects::EST_UInt:
					if (D3D_SVT_INT == stateAssignment.Type || D3D_SVT_UINT == stateAssignment.Type)
						reinterpret_cast<INT*>(data)[constantIdx] = constant.iValue;
					else if (D3D_SVT_UINT8 == stateAssignment.Type)
						reinterpret_cast<UINT8*>(data)[constantIdx] = static_cast<UINT8>(constant.iValue);
					else if (D3D_SVT_BOOL == stateAssignment.Type)
						reinterpret_cast<BOOL*>(data)[constantIdx] = static_cast<BOOL>(constant.iValue);
					else
						Check(false, D3DEFFECTSLITE_MAKE_LINE("State attribute type mismatch, expected integer"));
					break;
				case D3DX11Effects::EST_Bool:
					Check(D3D_SVT_BOOL == stateAssignment.Type, D3DEFFECTSLITE_MAKE_LINE("State attribute type mismatch, expected bool"));
					reinterpret_cast<BOOL*>(data)[constantIdx] = constant.bValue;
					break;
				}
			}
		}
		else
			Check(false, D3DEFFECTSLITE_MAKE_LINE("Unsupported assignment type"));
	}
}

void DefaultInitializeState(D3D11_RASTERIZER_DESC &info)
{
	info.FillMode = D3D11_FILL_SOLID;
	info.CullMode = D3D11_CULL_BACK;
	info.FrontCounterClockwise = FALSE;
	info.DepthBias = D3D11_DEFAULT_DEPTH_BIAS;
	info.DepthBiasClamp = D3D11_DEFAULT_DEPTH_BIAS_CLAMP;
	info.SlopeScaledDepthBias = D3D11_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
	info.DepthClipEnable = TRUE;
	info.ScissorEnable = FALSE;
	info.MultisampleEnable = FALSE;
	info.AntialiasedLineEnable = FALSE;
}

void DefaultInitializeState(D3D11_DEPTH_STENCIL_DESC &info)
{
	info.DepthEnable = TRUE;
	info.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	info.DepthFunc = D3D11_COMPARISON_LESS;
	info.StencilEnable = FALSE;
	info.StencilReadMask = D3D11_DEFAULT_STENCIL_READ_MASK;
	info.StencilWriteMask = D3D11_DEFAULT_STENCIL_WRITE_MASK;
	info.FrontFace.StencilFunc = info.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
	info.FrontFace.StencilDepthFailOp = info.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
	info.FrontFace.StencilPassOp = info.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	info.FrontFace.StencilFailOp = info.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
}

void DefaultInitializeState(D3D11_BLEND_DESC &info)
{
	info.AlphaToCoverageEnable = FALSE;
	info.IndependentBlendEnable = FALSE;
	for (int i = 0; i < arraylen(info.RenderTarget); ++i)
	{
		info.RenderTarget[i].BlendEnable = FALSE;
		info.RenderTarget[i].SrcBlend = D3D11_BLEND_ONE;
		info.RenderTarget[i].DestBlend = D3D11_BLEND_ZERO;
		info.RenderTarget[i].BlendOp = D3D11_BLEND_OP_ADD;
		info.RenderTarget[i].SrcBlendAlpha = D3D11_BLEND_ONE;
		info.RenderTarget[i].DestBlendAlpha = D3D11_BLEND_ZERO;
		info.RenderTarget[i].BlendOpAlpha = D3D11_BLEND_OP_ADD;
		info.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
	}
}

void DefaultInitializeState(D3D11_SAMPLER_DESC &info)
{
	info.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	info.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	info.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	info.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	info.MinLOD = -FLT_MAX;
	info.MaxLOD = FLT_MAX;
	info.MipLODBias = D3D11_DEFAULT_MIP_LOD_BIAS;
	info.MaxAnisotropy = D3D11_DEFAULT_MAX_ANISOTROPY;
	info.ComparisonFunc = D3D11_COMPARISON_NEVER;
	for (int i = 0; i < arraylen(info.BorderColor); ++i)
		info.BorderColor[i] = D3D11_DEFAULT_BORDER_COLOR_COMPONENT;
}

template <class StateInfo>
StateInfo* LoadState(Reader &reader, Reader &unstructuredReader, PlainReflection &reflection,
					 VariableInfo *pVariableInfo, const D3DX11Effects::SBinaryType &type, D3DX11Effects::EObjectType objType,
					 ArrayPairRef<StateInfo> state, AssignmentSelector &stateAssignmentSelector)
{
	Heap &heap = reflection.Heap;

	StateInfo *pFirstStateInfo = nullptr;
	UINT successorCount = max(type.Elements, 1U);

	while (successorCount-- > 0)
	{
		StateInfo *pStateInfo = AddAndMaybeConstruct(state);
		if (!pFirstStateInfo) pFirstStateInfo = pStateInfo;

		if (pStateInfo)
			DefaultInitializeState(pStateInfo->Desc);

		LoadStateAssignments(reader, unstructuredReader, pStateInfo, stateAssignmentSelector);

		if (pStateInfo)
		{
			pStateInfo->Variable = pVariableInfo;
			pStateInfo->SuccessorCount = successorCount;
		}
	}

	return pFirstStateInfo;
}

void LoadObjectVariables(Reader &reader, Reader &unstructuredReader, const D3DX11Effects::SBinaryHeader &header, PlainReflection &reflection, ScratchPool &scratchPool)
{
	EffectInfo &info = reflection.Info;
	Heap &heap = reflection.Heap;

	for (UINT variableIdx = 0; variableIdx < header.Effect.cObjectVariables; ++variableIdx)
	{
		const D3DX11Effects::SBinaryObjectVariable &variable = Read<D3DX11Effects::SBinaryObjectVariable>(reader);
		VariableInfo *pVariableInfo = AddAndMaybeConstruct(Variables(info));
		char *name = MoveString(heap, FetchMultiple<char>(unstructuredReader, variable.oName, 1));
		char *pSemantic = (variable.oSemantic != 0) ? MoveString(heap, FetchMultiple<char>(unstructuredReader, variable.oSemantic, 1)) : nullptr;
		
		D3D_SHADER_VARIABLE_TYPE variableType = D3D_SVT_VOID;
		void *untypedObjectInfo;

		unstructuredReader.pointer = variable.oType;
		const D3DX11Effects::SBinaryType &type = Read<D3DX11Effects::SBinaryType>(unstructuredReader);
		Check(type.VarType == D3DX11Effects::EVT_Object, D3DEFFECTSLITE_MAKE_LINE("Expected object type"));
		D3DX11Effects::EObjectType objType = Read<D3DX11Effects::EObjectType>(unstructuredReader);

		// Strings
		if (objType == D3DX11Effects::EOT_String)
		{
			variableType = D3D_SVT_STRING;
			untypedObjectInfo = LoadStringConstant(reader, unstructuredReader, reflection, pVariableInfo, type);
		}
		// Resources
		else if ((variableType = ToSRVType(objType)) != D3D_SVT_VOID)
			untypedObjectInfo = LoadResource(reflection, pVariableInfo, variable, type, Resources(info.Resources));
		else if ((variableType = ToUAVType(objType)) != D3D_SVT_VOID)
			untypedObjectInfo = LoadResource(reflection, pVariableInfo, variable, type, UAVs(info.Resources));
		else if ((variableType = ToRenderTargetType(objType)) != D3D_SVT_VOID)
			untypedObjectInfo = LoadResource(reflection, pVariableInfo, variable, type, RenderTargets(info.Resources));
		else if ((variableType = ToDepthStencilTargetType(objType)) != D3D_SVT_VOID)
			untypedObjectInfo = LoadResource(reflection, pVariableInfo, variable, type, DepthStencilTargets(info.Resources));
		// Shaders
		else if ((variableType = ToShaderType(objType)) != D3D_SVT_VOID)
			untypedObjectInfo = LoadShader(reader, unstructuredReader, reflection, scratchPool, pVariableInfo, type, objType, Shaders(info.Shaders, variableType));
		// State
		else if ((variableType = ToStateType(objType)) != D3D_SVT_VOID)
		{
			switch (variableType)
			{
			case D3D_SVT_SAMPLER:
				{
					SamplerStateInfo *pSamplerInfo = LoadState(reader, unstructuredReader, reflection, pVariableInfo, type, objType, SamplerState(info.State), SamplerAssignment);
					if (pSamplerInfo)
						pSamplerInfo->BindPoint = variable.ExplicitBindPoint;
					untypedObjectInfo = pSamplerInfo;
				}
				break;
			case D3D_SVT_BLEND:
				untypedObjectInfo = LoadState(reader, unstructuredReader, reflection, pVariableInfo, type, objType, BlendState(info.State), BlendAssignment);
				break;
			case D3D_SVT_DEPTHSTENCIL:
				untypedObjectInfo = LoadState(reader, unstructuredReader, reflection, pVariableInfo, type, objType, DepthStencilState(info.State), DepthStencilAssignment);
				break;
			case D3D_SVT_RASTERIZER:
				untypedObjectInfo = LoadState(reader, unstructuredReader, reflection, pVariableInfo, type, objType, RasterizerState(info.State), RasterizerAssignment);
				break;
			}
		}

		Check(variableType != D3D_SVT_VOID, D3DEFFECTSLITE_MAKE_LINE("Unsupported object type"));

		AnnotationBlockInfo *pAnnotations = LoadAnnotations(reader, unstructuredReader, heap, scratchPool.Annotations);

		if (pVariableInfo)
		{
			pVariableInfo->Name = name;
			pVariableInfo->Semantic = pSemantic;
			pVariableInfo->Annotations = pAnnotations;
			pVariableInfo->Type = variableType;
			pVariableInfo->Info = untypedObjectInfo;
		}
	}
}

struct BindAssignment
{
	D3DX11Effects::ELhsType Value;
	D3D_SHADER_VARIABLE_TYPE Type;
	UINT Elements;
	UINT IndexCount;
	UINT DataOffset;
	UINT DataStride;
};

const UINT PassAssignmentsBaseIndex = 0;
const BindAssignment PassAssignments[] =
{
	{ D3DX11Effects::ELHS_RasterizerBlock, D3D_SVT_RASTERIZER, 1, 1, offsetof_uint(RasterizerStateBindInfo, Block), 0 },
	{ D3DX11Effects::ELHS_DepthStencilBlock, D3D_SVT_DEPTHSTENCIL, 1, 1, offsetof_uint(DepthStencilStateBindInfo, Block), 0 },
	{ D3DX11Effects::ELHS_BlendBlock, D3D_SVT_BLEND, 1, 1, offsetof_uint(BlendStateBindInfo, Block), 0 },
	
	{ D3DX11Effects::ELHS_RenderTargetView, D3D_SVT_RENDERTARGETVIEW, 1, 8, offsetof_uint(RenderTargetBindInfo, RenderTargets), strideof_uint(RenderTargetBindInfo, RenderTargets) },
	{ D3DX11Effects::ELHS_DepthStencilView, D3D_SVT_DEPTHSTENCILVIEW, 1, 1, offsetof_uint(RenderTargetBindInfo, DepthStencilTarget), 0 },
	// MONITOR: GenerateMips? Unsupported ...
	{ D3DX11Effects::ELHS_GenerateMips, D3D_SVT_TEXTURE, 0, 0, 0, 0 },
	
	{ D3DX11Effects::ELHS_VertexShaderBlock, D3D_SVT_VERTEXSHADER, 1, 1, 0, 0 },
	{ D3DX11Effects::ELHS_PixelShaderBlock, D3D_SVT_PIXELSHADER, 1, 1, 0, 0 },
	{ D3DX11Effects::ELHS_GeometryShaderBlock, D3D_SVT_GEOMETRYSHADER, 1, 1, 0, 0 },
	
	{ D3DX11Effects::ELHS_DS_StencilRef, D3D_SVT_UINT, 1, 1, offsetof_uint(DepthStencilStateBindInfo, StencilRef), 0 },
	{ D3DX11Effects::ELHS_B_BlendFactor,  D3D_SVT_FLOAT, 4, 1, offsetof_uint(BlendStateBindInfo, BlendFactor), 0 },
	{ D3DX11Effects::ELHS_B_SampleMask, D3D_SVT_UINT, 1, 1, offsetof_uint(BlendStateBindInfo, SampleMask), 0 }
};

const UINT PassAssignments11BaseIndex = 56;
const BindAssignment PassAssignments11[] =
{
	{ D3DX11Effects::ELHS_HullShaderBlock, D3D_SVT_HULLSHADER, 1, 1, 0, 0 },
	{ D3DX11Effects::ELHS_DomainShaderBlock, D3D_SVT_DOMAINSHADER, 1, 1, 0, 0 },
	{ D3DX11Effects::ELHS_ComputeShaderBlock, D3D_SVT_COMPUTESHADER, 1, 1, 0, 0 },
};

const BindAssignment& PassAssignment(UINT index)
{
	if (PassAssignmentsBaseIndex <= index && index - PassAssignmentsBaseIndex < arraylen(PassAssignments))
	{
		index -= PassAssignmentsBaseIndex;
		Check(PassAssignments[index].Value != D3DX11Effects::ELHS_GenerateMips,
			D3DEFFECTSLITE_MAKE_LINE("Unsupported pass operation"));
		return PassAssignments[index];
	}
	else if(PassAssignments11BaseIndex <= index && index - PassAssignments11BaseIndex < arraylen(PassAssignments11))
		return PassAssignments11[index - PassAssignments11BaseIndex];
	else
		Check(false, D3DEFFECTSLITE_MAKE_LINE("Unsupported pass attribute")), D3DEFFECTSLITE_ASSUME(false);
}

typedef const BindAssignment& BindAssignmentSelector(UINT index);

bool IsShader(D3DX11Effects::ELhsType lhsType)
{
	switch (lhsType)
	{
	case D3DX11Effects::ELHS_VertexShaderBlock:
	case D3DX11Effects::ELHS_PixelShaderBlock:
	case D3DX11Effects::ELHS_GeometryShaderBlock:
//	case D3DX11Effects::ELHS_GeometryShaderSO: MONITOR: ???
	case D3DX11Effects::ELHS_ComputeShaderBlock:
	case D3DX11Effects::ELHS_HullShaderBlock:
	case D3DX11Effects::ELHS_DomainShaderBlock: return true;
	default: return false;
	}
}

ShaderInfo*& ShaderBlock(PassShaderBindingInfo &shaderSet, D3DX11Effects::ELhsType lhsType)
{
	switch (lhsType)
	{
	case D3DX11Effects::ELHS_VertexShaderBlock: return shaderSet.VertexShader;
	case D3DX11Effects::ELHS_PixelShaderBlock: return shaderSet.PixelShader;
	case D3DX11Effects::ELHS_GeometryShaderBlock: return shaderSet.GeometryShader;
//	case D3DX11Effects::ELHS_GeometryShaderSO: MONITOR:  ???
	case D3DX11Effects::ELHS_ComputeShaderBlock: return shaderSet.ComputeShader;
	case D3DX11Effects::ELHS_HullShaderBlock: return shaderSet.HullShader;
	case D3DX11Effects::ELHS_DomainShaderBlock: return shaderSet.DomainShader;
	default: assert(false); D3DEFFECTSLITE_ASSUME(false);
	}
}

void LoadShaderAssignment(Reader &unstructuredReader, PlainReflection &reflection, ScratchPool &scratchPool, ShaderInfo **ppShaderBlock,
						  const D3DX11Effects::SBinaryAssignment &assignment, const BindAssignment &passAssignment,
						  ArrayPairRef<ShaderInfo> shaders)
{
	Heap &heap = reflection.Heap;

	unstructuredReader.pointer = assignment.oInitializer;

	switch (assignment.AssignmentType)
	{
	case D3DX11Effects::ECAT_Constant:
		{
			Check(Read<UINT>(unstructuredReader) == 1, D3DEFFECTSLITE_MAKE_LINE("Expected shader NULL assignment"));
			Check(Read<D3DX11Effects::SBinaryConstant>(unstructuredReader).iValue == 0, D3DEFFECTSLITE_MAKE_LINE("Expected shader NULL assignment"));

			// NOTE: Shader already nulled above
		}
		break;
	case D3DX11Effects::ECAT_ConstIndex:
	case D3DX11Effects::ECAT_Variable:
		{
			const char *variableName;
			UINT variableElement = 0;

			if (assignment.AssignmentType == D3DX11Effects::ECAT_ConstIndex)
			{
				const D3DX11Effects::SBinaryAssignment::SConstantIndex &index = Read<D3DX11Effects::SBinaryAssignment::SConstantIndex>(unstructuredReader);
				variableName = FetchMultiple<char>(unstructuredReader, index.oArrayName, 1);
				variableElement = index.Index;
			}
			else
				variableName = ReadMultiple<char>(unstructuredReader, 1);

			if (ppShaderBlock)
			{
				VariableInfo* pVariableInfo = FindVariable(reflection.Info, variableName);

				Check(pVariableInfo != nullptr, D3DEFFECTSLITE_MAKE_LINE("Cannot find shader variable"));
				Check(pVariableInfo->Type == passAssignment.Type, D3DEFFECTSLITE_MAKE_LINE("Shader variable type mismatch"));

				*ppShaderBlock = static_cast<ShaderInfo*>(pVariableInfo->Info.get());

				Check(variableElement <= (*ppShaderBlock)->SuccessorCount, D3DEFFECTSLITE_MAKE_LINE("Shader variable array index out of bounds"));
				*ppShaderBlock += variableElement;
			}
			
		}
		break;
	case D3DX11Effects::ECAT_InlineShader:
	case D3DX11Effects::ECAT_InlineShader5:
		{
			UINT shaderOffsetCheck = Peek<UINT>(unstructuredReader);
			UINT byteCountCheck = Fetch<UINT>(unstructuredReader, shaderOffsetCheck);

			// Apparently, there are empty shaders?!
			if (byteCountCheck > 0)
			{
				ShaderInfo *pShaderInfo = AddAndMaybeConstruct(shaders);
				StreamOutInfo *pStreamOutInfo = nullptr;

				UINT shaderOffset;
				
				if (assignment.AssignmentType == D3DX11Effects::ECAT_InlineShader)
				{
					const D3DX11Effects::SBinaryAssignment::SInlineShader &shader = Read<D3DX11Effects::SBinaryAssignment::SInlineShader>(unstructuredReader);
					shaderOffset = shader.oShader;

					if (shader.oSODecl != 0)
						pStreamOutInfo = LoadStreamOut(unstructuredReader, reflection, scratchPool, pShaderInfo, 1, &shader.oSODecl, 0);
				}
				else
				{
					const D3DX11Effects::SBinaryShaderData5 &shader = Read<D3DX11Effects::SBinaryShaderData5>(unstructuredReader);
					shaderOffset = shader.oShader;

					if (shader.cSODecls > 0)
						pStreamOutInfo = LoadStreamOut(unstructuredReader, reflection, scratchPool, pShaderInfo, shader.cSODecls, shader.oSODecls, shader.RasterizedStream);
				}

				unstructuredReader.pointer = shaderOffset;
				UINT byteCount = Read<UINT>(unstructuredReader);

				assert(shaderOffset == shaderOffsetCheck);
				assert(byteCount == byteCountCheck);

				const void *inlineByteCode = ReadMultiple<BYTE>(unstructuredReader, byteCount);
				void *byteCode = MoveData(heap, inlineByteCode, byteCount);

				UINT shaderIdx = scratchPool.Shaders.Count++;
				scratchPool.Shaders.Array[shaderIdx].Shader = pShaderInfo;

				// Create reflection, if not yet existent
				if (scratchPool.Shaders.Array[shaderIdx].Reflection.get() == nullptr)
				{
					HRESULT reflectionResult = ::D3DReflect(inlineByteCode, byteCount, IID_ID3D11ShaderReflection, (void**)&scratchPool.Shaders.Array[shaderIdx].Reflection);
					Check(reflectionResult == S_OK, D3DEFFECTSLITE_MAKE_LINE("Cannot create shader reflection object"));
				}

				if (pShaderInfo)
				{
					pShaderInfo->Variable = nullptr;
					pShaderInfo->ByteCount = byteCount;
					pShaderInfo->ByteCode = byteCode;
					pShaderInfo->StreamOut = pStreamOutInfo;
					pShaderInfo->SuccessorCount = 0;
				}

				if (ppShaderBlock)
					*ppShaderBlock = pShaderInfo;
			}
		}
		break;
	default:
		Check(false, D3DEFFECTSLITE_MAKE_LINE("Unsupported shader assignment"));
		D3DEFFECTSLITE_ASSUME(false);
	}
}

D3DX11Effects::ELhsType ParentState(D3DX11Effects::ELhsType lhsType)
{
	switch (lhsType)
	{
	case D3DX11Effects::ELHS_RasterizerBlock: return D3DX11Effects::ELHS_RasterizerBlock;
	case D3DX11Effects::ELHS_DepthStencilBlock: return D3DX11Effects::ELHS_DepthStencilBlock;
	case D3DX11Effects::ELHS_BlendBlock: return D3DX11Effects::ELHS_BlendBlock;
	case D3DX11Effects::ELHS_DS_StencilRef: return D3DX11Effects::ELHS_DepthStencilBlock;
	case D3DX11Effects::ELHS_B_BlendFactor: return D3DX11Effects::ELHS_BlendBlock;
	case D3DX11Effects::ELHS_B_SampleMask: return D3DX11Effects::ELHS_BlendBlock;
	default: return D3DX11Effects::ELHS_Invalid;
	}
}

template <class StateInfo, class StateBindInfo>
void LoadStateAssignment(Reader &unstructuredReader, PlainReflection &reflection, StateBindInfo *&piStateBlock,
						 const D3DX11Effects::SBinaryAssignment &assignment, const BindAssignment &passAssignment)
{
	Heap &heap = reflection.Heap;

	if (!piStateBlock)
	{
		piStateBlock = heap.NoConstruct<StateBindInfo>();

		if (piStateBlock)
			memset(piStateBlock, 0, sizeof(*piStateBlock));
		else
			piStateBlock = invalidptr;
	}

	unstructuredReader.pointer = assignment.oInitializer;

	switch (assignment.AssignmentType)
	{
	case D3DX11Effects::ECAT_Constant:
		{
			if (IsObjectAssignmentHelper(passAssignment.Value))
			{
				Check(Read<UINT>(unstructuredReader) == 1, D3DEFFECTSLITE_MAKE_LINE("Expected state NULL assignment"));
				Check(Read<D3DX11Effects::SBinaryConstant>(unstructuredReader).iValue == 0, D3DEFFECTSLITE_MAKE_LINE("Expected state NULL assignment"));

				// NOTE: State already nulled above
			}
			else
			{
				Check(assignment.Index < passAssignment.IndexCount, D3DEFFECTSLITE_MAKE_LINE("State value index out of bounds"));
				
				UINT constantCount = Read<UINT>(unstructuredReader);
				const D3DX11Effects::SBinaryConstant* constants = ReadMultiple<D3DX11Effects::SBinaryConstant>(unstructuredReader, constantCount);

				Check(constantCount == passAssignment.Elements, D3DEFFECTSLITE_MAKE_LINE("State value element count mismatch"));

				if (ptrvalid(piStateBlock))
				{
					BYTE *data = reinterpret_cast<BYTE*>(piStateBlock) + passAssignment.DataOffset + (assignment.Index * passAssignment.DataStride);

					for (UINT constantIdx = 0; constantIdx < constantCount; ++constantIdx)
					{
						const D3DX11Effects::SBinaryConstant &constant = constants[constantIdx];

						switch (constant.Type)
						{
						case D3DX11Effects::EST_Float:
							Check(D3D_SVT_FLOAT == passAssignment.Type, D3DEFFECTSLITE_MAKE_LINE("State value type mismatch, expected float"));
							reinterpret_cast<FLOAT*>(data)[constantIdx] = constant.fValue;
							break;
						case D3DX11Effects::EST_Int:
						case D3DX11Effects::EST_UInt:
							if (D3D_SVT_INT == passAssignment.Type || D3D_SVT_UINT == passAssignment.Type)
								reinterpret_cast<INT*>(data)[constantIdx] = constant.iValue;
							else if (D3D_SVT_UINT8 == passAssignment.Type)
								reinterpret_cast<UINT8*>(data)[constantIdx] = static_cast<UINT8>(constant.iValue);
							else
								Check(false, D3DEFFECTSLITE_MAKE_LINE("State value type mismatch, expected integer"));
							break;
						case D3DX11Effects::EST_Bool:
							Check(D3D_SVT_BOOL == passAssignment.Type, D3DEFFECTSLITE_MAKE_LINE("State attribute type mismatch, expected bool"));
							reinterpret_cast<BOOL*>(data)[constantIdx] = constant.bValue;
							break;
						}
					}
				}
			}
		}
		break;
	case D3DX11Effects::ECAT_ConstIndex:
	case D3DX11Effects::ECAT_Variable:
		{
			Check(IsObjectAssignmentHelper(passAssignment.Value) != 0, D3DEFFECTSLITE_MAKE_LINE("State variable assignment only supported for state blocks"));

			const char *variableName;
			UINT variableElement = 0;

			if (assignment.AssignmentType == D3DX11Effects::ECAT_ConstIndex)
			{
				const D3DX11Effects::SBinaryAssignment::SConstantIndex &index = Read<D3DX11Effects::SBinaryAssignment::SConstantIndex>(unstructuredReader);
				variableName = FetchMultiple<char>(unstructuredReader, index.oArrayName, 1);
				variableElement = index.Index;
			}
			else
				variableName = ReadMultiple<char>(unstructuredReader, 1);

			if (ptrvalid(piStateBlock))
			{
				VariableInfo* pVariableInfo = FindVariable(reflection.Info, variableName);

				Check(pVariableInfo != nullptr, D3DEFFECTSLITE_MAKE_LINE("Cannot find state variable"));
				Check(pVariableInfo->Type == passAssignment.Type, D3DEFFECTSLITE_MAKE_LINE("State variable type mismatch"));

				piStateBlock->Block = static_cast<StateInfo*>(pVariableInfo->Info.get());

				Check(variableElement <= piStateBlock->Block->SuccessorCount, D3DEFFECTSLITE_MAKE_LINE("State variable array index out of bounds"));
				piStateBlock->Block += variableElement;
			}
		}
		break;
	default:
		Check(false, D3DEFFECTSLITE_MAKE_LINE("Unsupported state assignment"));
		D3DEFFECTSLITE_ASSUME(false);
	}
}

bool IsRenderTarget(D3DX11Effects::ELhsType lhsType)
{
	switch (lhsType)
	{
	case D3DX11Effects::ELHS_RenderTargetView:
	case D3DX11Effects::ELHS_DepthStencilView: return true;
	default: return false;
	}
}

void LoadRenderTargetAssignment(Reader &unstructuredReader, PlainReflection &reflection, RenderTargetBindInfo *&piRTBlock,
								const D3DX11Effects::SBinaryAssignment &assignment, const BindAssignment &passAssignment)
{
	Heap &heap = reflection.Heap;

	if (!piRTBlock)
	{
		piRTBlock = heap.NoConstruct<RenderTargetBindInfo>();

		if (piRTBlock)
			memset(piRTBlock, 0, sizeof(*piRTBlock));
		else
			piRTBlock = invalidptr;
	}

	unstructuredReader.pointer = assignment.oInitializer;

	switch (assignment.AssignmentType)
	{
	case D3DX11Effects::ECAT_Constant:
		{
			Check(Read<UINT>(unstructuredReader) == 1, D3DEFFECTSLITE_MAKE_LINE("Expected render target NULL assignment"));
			Check(Read<D3DX11Effects::SBinaryConstant>(unstructuredReader).iValue == 0, D3DEFFECTSLITE_MAKE_LINE("Expected render target NULL assignment"));
		}
		break;
	case D3DX11Effects::ECAT_ConstIndex:
	case D3DX11Effects::ECAT_Variable:
		{
			Check(assignment.Index < passAssignment.IndexCount, D3DEFFECTSLITE_MAKE_LINE("Render target index out of bounds"));
				
			UINT constantCount = Read<UINT>(unstructuredReader);
			const D3DX11Effects::SBinaryConstant* constants = ReadMultiple<D3DX11Effects::SBinaryConstant>(unstructuredReader, constantCount);

			Check(constantCount == passAssignment.Elements, D3DEFFECTSLITE_MAKE_LINE("Multiple render targets at a time?! Then go fix me!"));

			const char *variableName;
			UINT variableElement = 0;

			if (assignment.AssignmentType == D3DX11Effects::ECAT_ConstIndex)
			{
				const D3DX11Effects::SBinaryAssignment::SConstantIndex &index = Read<D3DX11Effects::SBinaryAssignment::SConstantIndex>(unstructuredReader);
				variableName = FetchMultiple<char>(unstructuredReader, index.oArrayName, 1);
				variableElement = index.Index;
			}
			else
				variableName = ReadMultiple<char>(unstructuredReader, 1);

			if (ptrvalid(piRTBlock))
			{
				VariableInfo* pVariableInfo = FindVariable(reflection.Info, variableName);

				Check(pVariableInfo != nullptr, D3DEFFECTSLITE_MAKE_LINE("Cannot find render target variable"));
				Check(pVariableInfo->Type == passAssignment.Type, D3DEFFECTSLITE_MAKE_LINE("Render target variable type mismatch"));

				ResourceInfo *&TargetSlot = *reinterpret_cast<ResourceInfo**>(
						reinterpret_cast<BYTE*>(piRTBlock) + passAssignment.DataOffset + (assignment.Index * passAssignment.DataStride)
					);
				TargetSlot = static_cast<ResourceInfo*>(pVariableInfo->Info.get());

				Check(variableElement <= TargetSlot->SuccessorCount, D3DEFFECTSLITE_MAKE_LINE("Render target variable array index out of bounds"));
				TargetSlot += variableElement;
			}
		}
		break;
	default:
		Check(false, D3DEFFECTSLITE_MAKE_LINE("Unsupported state assignment"));
		D3DEFFECTSLITE_ASSUME(false);
	}
}

PassInfo* LoadPasses(Reader &reader, Reader &unstructuredReader, PlainReflection &reflection, ScratchPool &scratchPool, UINT passCount)
{
	EffectInfo &info = reflection.Info;
	Heap &heap = reflection.Heap;

	PassInfo *pPasses = BasePointer(Passes(info));

	for (UINT variableIdx = 0; variableIdx < passCount; ++variableIdx)
	{
		const D3DX11Effects::SBinaryPass &pass = Read<D3DX11Effects::SBinaryPass>(reader);
		PassInfo *pPassInfo = AddAndMaybeConstruct(Passes(info));
		char *name = MoveString(heap, FetchMultiple<char>(unstructuredReader, pass.oName, 1));
		
		AnnotationBlockInfo *pAnnotations = LoadAnnotations(reader, unstructuredReader, heap, scratchPool.Annotations);
		
		if (pPassInfo)
		{
			pPassInfo->Name = name;
			pPassInfo->Annotations = pAnnotations;
		}

		// Load assignments
		const D3DX11Effects::SBinaryAssignment *assignments = ReadMultiple<D3DX11Effects::SBinaryAssignment>(reader, pass.cAssignments);

		PassShaderBindingInfo scratchShaders;
		PassStateBindingInfo scratchState;
		PassShaderBindingInfo &passShaders = (pPassInfo) ? pPassInfo->Shaders : scratchShaders;
		PassStateBindingInfo &passState = (pPassInfo) ? pPassInfo->State : scratchState;
		memset(&passShaders, 0, sizeof(passShaders));
		memset(&passState, 0, sizeof(passState));

		for (UINT assignmentIdx = 0; assignmentIdx < pass.cAssignments; ++assignmentIdx)
		{
			const D3DX11Effects::SBinaryAssignment &assignment = assignments[assignmentIdx];
			const BindAssignment &passAssignment = PassAssignment(assignment.iState);
			
			Check(assignment.Index < passAssignment.IndexCount, D3DEFFECTSLITE_MAKE_LINE("Pass attribute index out of bounds"));

			if (IsShader(passAssignment.Value))
				LoadShaderAssignment(unstructuredReader, reflection, scratchPool,
					(pPassInfo) ? &ShaderBlock(passShaders, passAssignment.Value) : nullptr,
					assignment, passAssignment, Shaders(info.Shaders, passAssignment.Type));
			else if (D3DX11Effects::ELhsType parentState = ParentState(passAssignment.Value))
				switch(parentState)
				{
				case D3DX11Effects::ELHS_RasterizerBlock:
					LoadStateAssignment<RasterizerStateInfo>(unstructuredReader, reflection, passState.RasterizerState.get(), assignment, passAssignment);
					break;
				case D3DX11Effects::ELHS_DepthStencilBlock:
					LoadStateAssignment<DepthStencilStateInfo>(unstructuredReader, reflection, passState.DepthStencilState.get(), assignment, passAssignment);
					break;
				case D3DX11Effects::ELHS_BlendBlock:
					LoadStateAssignment<BlendStateInfo>(unstructuredReader, reflection, passState.BlendState.get(), assignment, passAssignment);
					break;
				}
			else if (IsRenderTarget(passAssignment.Value))
				LoadRenderTargetAssignment(unstructuredReader, reflection, passState.Targets, assignment, passAssignment);
			else
				assert(false);
		}
	}

	return pPasses;
}

TechniqueInfo* LoadTechniques(Reader &reader, Reader &unstructuredReader, PlainReflection &reflection, ScratchPool &scratchPool, UINT techniqueCount)
{
	EffectInfo &info = reflection.Info;
	Heap &heap = reflection.Heap;

	TechniqueInfo *pTechniques = BasePointer(Techniques(info));

	for (UINT variableIdx = 0; variableIdx < techniqueCount; ++variableIdx)
	{
		const D3DX11Effects::SBinaryTechnique &technique = Read<D3DX11Effects::SBinaryTechnique>(reader);
		TechniqueInfo *pTechniqueInfo = AddAndMaybeConstruct(Techniques(info));
		char *name = MoveString(heap, FetchMultiple<char>(unstructuredReader, technique.oName, 1));
		
		AnnotationBlockInfo *pAnnotations = LoadAnnotations(reader, unstructuredReader, heap, scratchPool.Annotations);
		PassInfo *pPasses = LoadPasses(reader, unstructuredReader, reflection, scratchPool, technique.cPasses);

		if (pTechniqueInfo)
		{
			pTechniqueInfo->Name = name;
			pTechniqueInfo->Annotations = pAnnotations;
			pTechniqueInfo->PassCount = technique.cPasses;
			pTechniqueInfo->Passes = pPasses;
		}
	}

	return pTechniques;
}

void LoadGroups(Reader &reader, Reader &unstructuredReader, const D3DX11Effects::SBinaryHeader &header, const D3DX11Effects::SBinaryHeader5 *pHeader5,
				PlainReflection &reflection, ScratchPool &scratchPool)
{
	// Groups unsupported before FX 5.0
	if (!pHeader5)
	{
		LoadTechniques(reader, unstructuredReader, reflection, scratchPool, header.cTechniques);
		return;
	}

	EffectInfo &info = reflection.Info;
	Heap &heap = reflection.Heap;

	for (UINT variableIdx = 0; variableIdx < pHeader5->cGroups; ++variableIdx)
	{
		const D3DX11Effects::SBinaryGroup &group = Read<D3DX11Effects::SBinaryGroup>(reader);
		// NOTE: Don't add null group
		char *pName = MoveString(heap, (group.oName != 0) ? FetchMultiple<char>(unstructuredReader, group.oName, 1) : "$Ungrouped");
		GroupInfo *pGroupInfo = AddAndMaybeConstruct(Groups(info));
		
		AnnotationBlockInfo *pAnnotations = LoadAnnotations(reader, unstructuredReader, heap, scratchPool.Annotations);
		TechniqueInfo *pTechniques = LoadTechniques(reader, unstructuredReader, reflection, scratchPool, group.cTechniques);

		if (pGroupInfo)
		{
			pGroupInfo->Name = pName;
			pGroupInfo->Annotations = pAnnotations;
			pGroupInfo->TechniqueCount = group.cTechniques;
			pGroupInfo->Techniques = pTechniques;
		}
	}
}

static const UINT BindPointSignMask = (1 << 31);

template <class ResourceInfo>
void MarkExplicitBindPoints(ArrayPairRef<ResourceInfo> resources)
{
	if (!resources.Array)
		return;

	for (UINT i = 0; i < resources.Count; ++i)
	{
		ResourceInfo &resource = resources.Array[i];
		if ((resource.BindPoint & BindPointSignMask) == 0)
			resource.BindPoint |= BindPointSignMask;
	}
}

bool IsBindPointInitialized(UINT bindPoint)
{
	return ((bindPoint & BindPointSignMask) == 0);
}

static const UINT BindPointDetectedAmbiguous = BindPointSignMask - 1;

UINT CheckedBindPoint(UINT bindPoint, UINT newBindPoint)
{
	if (IsBindPointInitialized(bindPoint))
		return (newBindPoint == bindPoint) ? bindPoint : BindPointDetectedAmbiguous;
	else
		return newBindPoint;
}

template <class ResourceInfo>
void FixBindPoints(ArrayPairRef<ResourceInfo> resources)
{
	if (!resources.Array)
		return;

	for (UINT i = 0; i < resources.Count; ++i)
	{
		ResourceInfo &resource = resources.Array[i];

		if (IsBindPointInitialized(resource.BindPoint))
			resource.BindPoint = (resource.BindPoint != BindPointDetectedAmbiguous) ? resource.BindPoint : BindPointAmbiguous;
		else
			resource.BindPoint = (resource.BindPoint != BindPointAmbiguous) ? (resource.BindPoint ^ BindPointSignMask) : BindPointAmbiguous;
	}
}

void LoadShaderBindings(Reader &reader, Reader &unstructuredReader, const D3DX11Effects::SBinaryHeader &header, const D3DX11Effects::SBinaryHeader5 *pHeader5,
						PlainReflection &reflection, ScratchPool &scratchPool)
{
	EffectInfo &info = reflection.Info;
	Heap &heap = reflection.Heap;
	
	MarkExplicitBindPoints(ConstantBuffers(info.Resources));
	MarkExplicitBindPoints(TextureBuffers(info.Resources));
	MarkExplicitBindPoints(Resources(info.Resources));
	MarkExplicitBindPoints(UAVs(info.Resources));
	MarkExplicitBindPoints(SamplerState(info.State));

	for (UINT shaderIdx = 0; shaderIdx < scratchPool.Shaders.Count; ++shaderIdx)
	{
		ShaderInfo *pShaderInfo = scratchPool.Shaders.Array[shaderIdx].Shader;
		ID3D11ShaderReflection *pShaderReflection = scratchPool.Shaders.Array[shaderIdx].Reflection.get();

		assert(pShaderReflection != nullptr);

		D3D11_SHADER_DESC shaderDesc;
		Check(pShaderReflection->GetDesc(&shaderDesc) == S_OK, D3DEFFECTSLITE_MAKE_LINE("Unable to retrieve shader description"));

		ConstantBufferBindInfo *pConstantBuffersBase = BasePointer(ConstantBuffers(scratchPool.ShaderBindings));
		ConstantBufferBindInfo *pTextureBuffersBase = BasePointer(TextureBuffers(scratchPool.ShaderBindings));
		ResourceBindInfo *pResourcesBase = BasePointer(Resources(scratchPool.ShaderBindings));
		ResourceBindInfo *pUAVsBase = BasePointer(UAVs(scratchPool.ShaderBindings));
		SamplerStateBindInfo *pSamplersBase = BasePointer(Samplers(scratchPool.ShaderBindings));

		// Resources
		for (UINT resourceIdx = 0; resourceIdx < shaderDesc.BoundResources; ++resourceIdx)
		{
			D3D11_SHADER_INPUT_BIND_DESC bindingDesc;
			Check(pShaderReflection->GetResourceBindingDesc(resourceIdx, &bindingDesc) == S_OK, D3DEFFECTSLITE_MAKE_LINE("Unable to retrieve resource binding description"));

			UINT startIndex = 0;
			VariableInfo *pVariableInfo = FindVariable(reflection.Info, bindingDesc.Name, &startIndex);
			
			Check(!pShaderInfo || pVariableInfo, D3DEFFECTSLITE_MAKE_LINE("Unable to find bound variable"));

			bool isTBuffer = false;
			bool isSRV = false;

			switch (bindingDesc.Type)
			{
		//	case TBuffer || CBuffer:
				{
			case D3D_SIT_TBUFFER:
					isTBuffer = true;
			case D3D_SIT_CBUFFER:
				
					ConstantBufferBindInfo *pBindInfo = AddAndMaybeConstruct(
						(isTBuffer) ? TextureBuffers(scratchPool.ShaderBindings) : ConstantBuffers(scratchPool.ShaderBindings) );

					Check(startIndex == 0 && bindingDesc.BindCount == 1, D3DEFFECTSLITE_MAKE_LINE("Constant buffer arrays not supported"));

					if (pBindInfo)
					{
						Check(!isTBuffer && pVariableInfo->Type == D3D_SVT_CBUFFER || isTBuffer && pVariableInfo->Type == D3D_SVT_TBUFFER,
							D3DEFFECTSLITE_MAKE_LINE("Constant buffer binding type mismatch"));

						pBindInfo->Resource = static_cast<ConstantBufferInfo*>(pVariableInfo->Info.get());
						pBindInfo->Register = bindingDesc.BindPoint;

						// Find globally unambiguous bind points
						pBindInfo->Resource->BindPoint = CheckedBindPoint(pBindInfo->Resource->BindPoint, pBindInfo->Register);
					}
				}
				break;

		//	case Resource || UAV:
				{
			case D3D_SIT_TEXTURE: 
			case D3D_SIT_STRUCTURED:
			case D3D_SIT_BYTEADDRESS:
					isSRV = true;
			case D3D_SIT_UAV_RWTYPED:
			case D3D_SIT_UAV_RWSTRUCTURED:
			case D3D_SIT_UAV_RWBYTEADDRESS:
			case D3D_SIT_UAV_APPEND_STRUCTURED:
			case D3D_SIT_UAV_CONSUME_STRUCTURED:
			case D3D_SIT_UAV_RWSTRUCTURED_WITH_COUNTER:

					ResourceBindInfo *pBindInfo = AddAndMaybeConstructMultiple(
						(isSRV) ? Resources(scratchPool.ShaderBindings) : UAVs(scratchPool.ShaderBindings), bindingDesc.BindCount);

					if (pBindInfo)
					{
						ResourceInfo *pResourceInfo = static_cast<ResourceInfo*>(pVariableInfo->Info.get());
						Check(startIndex + bindingDesc.BindCount <= pResourceInfo->SuccessorCount + 1, D3DEFFECTSLITE_MAKE_LINE("Resource binding out of bounds"));
						pResourceInfo += startIndex;

						for (UINT i = 0; i < bindingDesc.BindCount; ++i)
						{
//							Check(!isTBuffer && pVariableInfo->Type == D3D_SVT_CBUFFER || isTBuffer && pVariableInfo->Type == D3D_SVT_TBUFFER,
//								D3DEFFECTSLITE_MAKE_LINE("Constant buffer binding type mismatch"));

							pBindInfo[i].Resource = pResourceInfo + i;
							pBindInfo[i].Register = bindingDesc.BindPoint + i;
						}

						// Find globally unambiguous bind points
						pBindInfo->Resource->BindPoint = CheckedBindPoint(pBindInfo->Resource->BindPoint, pBindInfo->Register);
					}
				}
				break;

			case D3D_SIT_SAMPLER:
				{
					SamplerStateBindInfo *pBindInfo = AddAndMaybeConstructMultiple(Samplers(scratchPool.ShaderBindings), bindingDesc.BindCount);

					if (pBindInfo)
					{
						SamplerStateInfo *pSamplerInfo = static_cast<SamplerStateInfo*>(pVariableInfo->Info.get());
						Check(startIndex + bindingDesc.BindCount <= pSamplerInfo->SuccessorCount + 1, D3DEFFECTSLITE_MAKE_LINE("Sampler binding out of bounds"));
						pSamplerInfo += startIndex;

						for (UINT i = 0; i < bindingDesc.BindCount; ++i)
						{
//							Check(!isTBuffer && pVariableInfo->Type == D3D_SVT_CBUFFER || isTBuffer && pVariableInfo->Type == D3D_SVT_TBUFFER,
//								D3DEFFECTSLITE_MAKE_LINE("Constant buffer binding type mismatch"));

							pBindInfo[i].Sampler = pSamplerInfo + i;
							pBindInfo[i].Register = bindingDesc.BindPoint + i;
						}

						// Find globally unambiguous bind points
						pBindInfo->Sampler->BindPoint = CheckedBindPoint(pBindInfo->Sampler->BindPoint, pBindInfo->Register);
					}
				}
				break;
			
			default:
				D3DEFFECTSLITE_LOG_LINE("Unsupported resource binding, ignored");
			}
		}

		ShaderBindingInfo *pBindingInfo = heap.NoConstruct<ShaderBindingInfo>();

		if (pBindingInfo)
		{
			ConstantBuffers(*pBindingInfo) = Range(ConstantBuffers(scratchPool.ShaderBindings), pConstantBuffersBase);
			TextureBuffers(*pBindingInfo) = Range(TextureBuffers(scratchPool.ShaderBindings), pTextureBuffersBase);
			Resources(*pBindingInfo) = Range(Resources(scratchPool.ShaderBindings), pResourcesBase);
			UAVs(*pBindingInfo) = Range(UAVs(scratchPool.ShaderBindings), pUAVsBase);
			Samplers(*pBindingInfo) = Range(Samplers(scratchPool.ShaderBindings), pSamplersBase);
		}

		if (pShaderInfo)
			pShaderInfo->Bindings = pBindingInfo;
	}

	FixBindPoints(ConstantBuffers(info.Resources));
	FixBindPoints(TextureBuffers(info.Resources));
	FixBindPoints(Resources(info.Resources));
	FixBindPoints(UAVs(info.Resources));
	FixBindPoints(SamplerState(info.State));
}

void AllocateConstants(PlainReflection &reflection)
{
	EffectInfo &info = reflection.Info;
	Heap &heap = reflection.Heap;

	NoConstructMultiple(heap, Floats(info.Constants));
	NoConstructMultiple(heap, Ints(info.Constants));
	NoConstructMultiple(heap, Bools(info.Constants));
	NoConstructMultiple(heap, Structs(info.Constants));
	NoConstructMultiple(heap, Strings(info.Constants));
}

void AllocateResources(PlainReflection &reflection)
{
	EffectInfo &info = reflection.Info;
	Heap &heap = reflection.Heap;

	NoConstructMultiple(heap, ConstantBuffers(info.Resources));
	NoConstructMultiple(heap, TextureBuffers(info.Resources));
	NoConstructMultiple(heap, Resources(info.Resources));
	NoConstructMultiple(heap, UAVs(info.Resources));
	NoConstructMultiple(heap, RenderTargets(info.Resources));
	NoConstructMultiple(heap, DepthStencilTargets(info.Resources));
}

void AllocateShaders(PlainReflection &reflection)
{
	EffectInfo &info = reflection.Info;
	Heap &heap = reflection.Heap;

	NoConstructMultiple(heap, VertexShaders(info.Shaders));
	NoConstructMultiple(heap, PixelShaders(info.Shaders));
	NoConstructMultiple(heap, GeometryShaders(info.Shaders));
	NoConstructMultiple(heap, ComputeShaders(info.Shaders));
	NoConstructMultiple(heap, HullShaders(info.Shaders));
	NoConstructMultiple(heap, DomainShaders(info.Shaders));
	NoConstructMultiple(heap, StreamOut(info.Shaders));
}

void AllocateState(PlainReflection &reflection)
{
	EffectInfo &info = reflection.Info;
	Heap &heap = reflection.Heap;

	NoConstructMultiple(heap, SamplerState(info.State));
	NoConstructMultiple(heap, RasterizerState(info.State));
	NoConstructMultiple(heap, DepthStencilState(info.State));
	NoConstructMultiple(heap, BlendState(info.State));
}

void AllocatePasses(PlainReflection &reflection)
{
	EffectInfo &info = reflection.Info;
	Heap &heap = reflection.Heap;

	NoConstructMultiple(heap, Groups(info));
	NoConstructMultiple(heap, Techniques(info));
	NoConstructMultiple(heap, Passes(info));
}

void AllocateGlobalVariables(PlainReflection &reflection)
{
	EffectInfo &info = reflection.Info;
	Heap &heap = reflection.Heap;

	// These variables are spread all over the place ...
	info.Variables = heap.NoConstructMultiple<VariableInfo>(info.VariableCount);

	AllocateConstants(reflection);
	AllocateResources(reflection);
	AllocateShaders(reflection);
	AllocateState(reflection);
	AllocatePasses(reflection);
}

/// Allocates global variables.
void AllocatePooledBindings(PlainReflection &reflection, ScratchPool &scratchPool)
{
	Heap &heap = reflection.Heap;

	NoConstructMultiple(heap, ConstantBuffers(scratchPool.ShaderBindings));
	NoConstructMultiple(heap, TextureBuffers(scratchPool.ShaderBindings));
	NoConstructMultiple(heap, Resources(scratchPool.ShaderBindings));
	NoConstructMultiple(heap, Samplers(scratchPool.ShaderBindings));
	NoConstructMultiple(heap, UAVs(scratchPool.ShaderBindings));
}

/// Allocates global variables.
void AllocatePooledAnnotations(PlainReflection &reflection, ScratchPool &scratchPool)
{
	Heap &heap = reflection.Heap;

	NoConstructMultiple(heap, Floats(scratchPool.Annotations));
	NoConstructMultiple(heap, Ints(scratchPool.Annotations));
	NoConstructMultiple(heap, Bools(scratchPool.Annotations));
	NoConstructMultiple(heap, Strings(scratchPool.Annotations));
}

/// Allocats stream out elements.
void AllocatePooledStreamOut(PlainReflection &reflection, ScratchPool &scratchPool)
{
	Heap &heap = reflection.Heap;

	NoConstructMultiple(heap, Elements(scratchPool.StreamOut));
	NoConstructMultiple(heap, BufferStrides(scratchPool.StreamOut));
}


/// Allocates global variables.
void AllocateScratch(PlainReflection &reflection, ScratchPool &scratchPool)
{
	AllocatePooledBindings(reflection, scratchPool);
	AllocatePooledAnnotations(reflection, scratchPool);
	AllocatePooledStreamOut(reflection, scratchPool);
}

/// Resets the counts in all global variables.
void ResetGlobalCounts(EffectInfo &info, ScratchPool &scratchPool)
{
	memset(&info, 0, D3DEL_EFFECT_INFO_COUNT_BLOCK);
	memset(&info.Constants, 0, D3DEL_EFFECT_CONSTANT_INFO_COUNT_BLOCK);
	memset(&info.Resources, 0, D3DEL_EFFECT_RESOURCE_INFO_COUNT_BLOCK);
	memset(&info.State, 0, D3DEL_EFFECT_STATE_INFO_COUNT_BLOCK);
	memset(&info.Shaders, 0, D3DEL_EFFECT_SHADER_INFO_COUNT_BLOCK);

	memset(&scratchPool.Annotations, 0, D3DEL_ANNOTATION_BLOCK_INFO_COUNT_BLOCK);
	memset(&scratchPool.ShaderBindings, 0, D3DEL_SHADER_BINDING_INFO_COUNT_BLOCK);
	scratchPool.Shaders.Count = 0;
	scratchPool.StreamOut.BufferCount = 0;
	scratchPool.StreamOut.ElementCount = 0;
}

} // namespace

// Creates an effect reflection interface.
Reflection* D3DEFFECTSLITE_STDCALL ReflectEffect(const void *bytes, UINT byteCount, Allocator *pScratchAllocator)
{
	assert(bytes);
	assert(byteCount);

	if (!pScratchAllocator)
		pScratchAllocator = GetGlobalAllocator();

	try
	{
		Reader reader(bytes, byteCount);

		com_ptr<PlainReflection> reflection = CreateEmptyPlainReflection();
		EffectInfo &info = reflection->Info;

		// Read header & check version
		const D3DX11Effects::SBinaryHeader &header = Peek<D3DX11Effects::SBinaryHeader>(reader);
		DWORD effectVersion = GetEffectVersion(header.Tag);

		const D3DX11Effects::SBinaryHeader5 *pHeader5 = nullptr;

		if (effectVersion >= D3D10_FXL_VERSION(5,0))
			pHeader5 = &Read<D3DX11Effects::SBinaryHeader5>(reader);
		else
			Read<D3DX11Effects::SBinaryHeader>(reader);
		
		// Deprecated features
		Check(!header.RequiresPool(), D3DEFFECTSLITE_MAKE_LINE("EffectPools not supported"));

		// Strange check performed by the original FX framework
		Check(header.cInlineShaders <= header.cTotalShaders, D3DEFFECTSLITE_MAKE_LINE("Invalid effect header: |inline shaders| > |total shaders|"));

		// Skip to structured memory
		Reader unstructuredReader(ReadMultiple<BYTE>(reader, header.cbUnstructured), header.cbUnstructured);
		reader.base += reader.pointer;

		ScratchPool scratchPool;

		// Allocate scratch shader storage
		scratchPool.Shaders.Array = AllocatorNewMultiple<ShaderAux>(*pScratchAllocator, header.cTotalShaders);
		scratchPool.Shaders.Allocator = pScratchAllocator;
		scratchPool.Shaders.AllocationCount = header.cTotalShaders;
		
		// Load or preallocate
		for (bool bPreAllocation = true; ; bPreAllocation = false)
		{
			// Rewind
			reader.pointer = 0;
			unstructuredReader.pointer = 0;

			// Reset global variables globally
			ResetGlobalCounts(info, scratchPool);

			// IMPORTANT: Load everything IN ORDER
			LoadConstantBuffers(reader, unstructuredReader, header, *reflection, scratchPool);
			LoadObjectVariables(reader, unstructuredReader, header, *reflection, scratchPool);
			// TODO: Interfaces
			LoadGroups(reader, unstructuredReader, header, pHeader5, *reflection, scratchPool);
			LoadShaderBindings(reader, unstructuredReader, header, pHeader5, *reflection, scratchPool);

			// Preallocation complete
			if (bPreAllocation)
			{
				// ORDER: Globally reserve space for global variables
				AllocateGlobalVariables(*reflection);
				AllocateScratch(*reflection, scratchPool);

				// ORDER: Allocate space for everything in one block of memory
				reflection->Heap.Allocate();

				// ORDER: Allocate globally BEFORE resetting counts!
				AllocateGlobalVariables(*reflection);
				AllocateScratch(*reflection, scratchPool);
			}
			// Loading complete
			else
				break;
		}

		return reflection.unbind();
	}
	catch (const ReaderUnderflow&)
	{
		D3DEFFECTSLITE_LOG_LINE("Buffer too small");
	}
	catch (const UnknownVersionError&)
	{
		D3DEFFECTSLITE_LOG_LINE("Unknown effect version");
	}
	catch (const CheckError &e)
	{
		Log(e.msg);
	}
	catch (...)
	{
		D3DEFFECTSLITE_LOG_LINE("Unknown error");
	}

	return nullptr;
}

} // namespace

// Creates an effect reflection interface.
D3DEffectsLiteReflection* D3DEFFECTSLITE_STDCALL D3DELReflectEffect(const void *bytes, UINT byteCount, D3DEffectsLiteAllocator *pScratchAllocator)
{
	return D3DEffectsLite::ReflectEffect(bytes, byteCount, pScratchAllocator);
}
