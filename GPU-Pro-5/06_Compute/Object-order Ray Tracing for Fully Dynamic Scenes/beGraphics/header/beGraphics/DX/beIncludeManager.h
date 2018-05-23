/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_INCLUDEMANAGER_DX
#define BE_GRAPHICS_INCLUDEMANAGER_DX

#include "beGraphics.h"
#include <D3DCommon.h>
#include <lean/tags/noncopyable.h>
#include <list>

#include <beCore/bePathResolver.h>
#include <beCore/beContentProvider.h>

namespace beGraphics
{

namespace DX
{

/// Include tracker interface.
class IncludeTracker
{
protected:
	~IncludeTracker() throw() { }

public:
	/// Called for each actual include file, passing the resolved file & revision.
	virtual void Track(const utf8_ntri &file) = 0;
};

/// Implements a Direct3D include handler.
class IncludeManager : public lean::nonassignable, public ID3DInclude
{
private:
	beCore::PathResolver *const m_resolver;
	beCore::ContentProvider *const m_provider;

	IncludeTracker *const m_pTracker;
	IncludeTracker *const m_pRawTracker;

	typedef std::list< lean::com_ptr<beCore::Content> > content_list;
	content_list m_openContent;

public:
	/// Constructor.
	IncludeManager(beCore::PathResolver &resolver, beCore::ContentProvider &provider,
			IncludeTracker *pTracker = nullptr, IncludeTracker *pRawTracker = nullptr)
		: m_resolver(&resolver),
		m_provider(&provider),
		m_pTracker(pTracker),
		m_pRawTracker(pRawTracker)
	{
	}

	/// Opens the given include file, returning its contents.
	STDMETHOD(Open)(D3D_INCLUDE_TYPE includeType, LPCSTR fileName, LPCVOID parentData, LPCVOID *ppData, UINT *pBytes)
	{
		LEAN_ASSERT(ppData && pBytes);

		try
		{
			// Track raw includes
			if (m_pRawTracker)
				m_pRawTracker->Track(fileName);

			// Get absolute path
			beCore::Exchange::utf8_string path = m_resolver->Resolve(fileName, true);

			// Track resolved includes
			if (m_pTracker)
				m_pTracker->Track(path);

			// Try to open include
			lean::com_ptr<beCore::Content> pContent = m_provider->GetContent(path);

			*ppData = pContent->Data();
			*pBytes = static_cast<UINT>( pContent->Size() );

			// Keep include open until Close()-call
			m_openContent.push_back( pContent.transfer() );
		}
		catch (...)
		{
			return E_FAIL;
		}

		return S_OK;
	}

	/// Closes the given include file.
    STDMETHOD(Close)(LPCVOID pData)
	{
		try
		{
			// Find matching include and close content
			for (content_list::iterator it = m_openContent.begin(); it != m_openContent.end(); ++it)
				if ((*it)->Data() == pData)
				{
					m_openContent.erase(it);
					break;
				}
		}
		catch (...)
		{
			return E_FAIL;
		}

		return S_OK;
	}

	/// Gets the underlying path resolver.
	LEAN_INLINE beCore::PathResolver& PathResolver() const { return *m_resolver; }
	/// Gets the underlying content provider.
	LEAN_INLINE beCore::ContentProvider& ContentProvider() const { return *m_provider; }
};

} // namespace

} // namespace

#endif