#define SAFE_RELEASE(x) if(x) { (x)->Release(); (x) = NULL; }

#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS
#include "tbb/include/tbb/tbb.h"

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>

#include <algorithm>
#include <string>
#include <vector>
#include <hash_map>

#include <d3d11.h>
#include <d3dx11.h>
#include <D3Dcompiler.h>
#include <d3dx9math.h>

#include "../Math/Math.h"
