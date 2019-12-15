#pragma once

#include <vector>

class CIntrusiveUnorderedSetItemHandle
{
public:
    CIntrusiveUnorderedSetItemHandle() : m_index( -1 ) { }
    bool IsInserted() const { return m_index >= 0; }

protected:
    int m_index;
};

template< class CContainer, class CItem, CIntrusiveUnorderedSetItemHandle& (CItem::*GetHandle)() >
class CIntrusiveUnorderedPtrSet
{
    struct SHandleSetter : public CIntrusiveUnorderedSetItemHandle
    {
        void Set(int i) { m_index = i; }
        int Get() const { return m_index; }
    };

public:
    void Add( CItem* pItem, bool mayBeAlreadyInserted = false )
    {
        SHandleSetter& handle = static_cast< SHandleSetter& >( ( pItem->*GetHandle )() );
        CContainer& container = *static_cast< CContainer* >( this );
        if( handle.IsInserted() )
        {
            _ASSERT( mayBeAlreadyInserted && "Trying to insert a pointer that was already inserted into the set!" );
        }
        else
        {
            handle.Set( container.size() );
            container.push_back( pItem );
        }
    }
    void Remove( CItem* pItem, bool mayBeNotInserted = false )
    {
        SHandleSetter& handle = static_cast< SHandleSetter& >( ( pItem->*GetHandle )() );
        CContainer& container = *static_cast< CContainer* >( this );
        if( handle.IsInserted() )
        {
            CItem* pLastItem = container.back();
            ( pLastItem->*GetHandle )() = handle;
            container[ handle.Get() ] = pLastItem;
            container.pop_back();
            handle.Set( -1 );
        }
        else
        {
            _ASSERT( mayBeNotInserted && "Trying to remove a pointer that was not inserted into the set!" );
        }
    }
};

template< class CItem, CIntrusiveUnorderedSetItemHandle& (CItem::*GetHandle)() >
class CVectorBasedIntrusiveUnorderedPtrSet :
    public std::vector< CItem* >,
    public CIntrusiveUnorderedPtrSet< CVectorBasedIntrusiveUnorderedPtrSet< CItem, GetHandle >, CItem, GetHandle >
{
};
