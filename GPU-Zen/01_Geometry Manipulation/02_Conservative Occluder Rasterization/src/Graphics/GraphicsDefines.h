#pragma once

#include <stdio.h>
#include <cassert>
#include <Windows.h>
#include <d3dx12.h>

#define SAFE_DELETE( ptr ) if ( ptr != nullptr ) { delete ptr; ptr = nullptr; }
#define SAFE_DELETE_ARRAY( ptr ) if ( ptr != nullptr ) { delete[] ptr; ptr = nullptr; }
#define SAFE_RELEASE( ptr ) if ( ptr != nullptr ) { ptr->Release(); ptr = nullptr; }
#define SAFE_RELEASE_UNMAP( ptr ) if ( ptr != nullptr ) { ptr->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) ); ptr->Release(); ptr = nullptr; }
#define SAFE_CLOSE( handle ) if ( handle != nullptr ) { CloseHandle( handle ); handle = nullptr; }

#ifdef _DEBUG
#define LOG( message, ... ) printf( message, __VA_ARGS__ )
#define HR( hr ) if ( FAILED( hr ) ) { throw; }
#else
#define LOG( message, ... )
#define HR( hr ) hr
#endif

typedef UINT Index;