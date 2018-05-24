# TileBasedShadows Makefile
# Nikita Kindt - 27th Feb. 2015
# Demo tested on Mint, Antergos and Gentoo

extf=./../../
incl=-I./../includes -I/usr/include/SDL2 -I$(extf)External/AntTweakBar/includes

proj_files=Aabb.cpp Application.cpp Camera.cpp Demo.cpp DemoMesh.cpp FileManager.cpp FinalProcessor.cpp Font.cpp \
    Frustum.cpp Image.cpp InputManager.cpp main.cpp Material.cpp OGL_BlendState.cpp OGL_DepthStencilState.cpp \
    OGL_IndexBuffer.cpp OGL_RasterizerState.cpp OGL_Renderer.cpp OGL_RenderTarget.cpp OGL_Sampler.cpp OGL_Shader.cpp \
    OGL_StructuredBuffer.cpp OGL_Texture.cpp OGL_TimerQueryObject.cpp OGL_UniformBuffer.cpp OGL_VertexBuffer.cpp \
    OGL_VertexLayout.cpp PathPointLight.cpp PointLight.cpp ResourceManager.cpp Sky.cpp stdafx.cpp TiledDeferred.cpp \
    TileMap.cpp TimeManager.cpp Window.cpp
obj_files=$(patsubst %.cpp, %.o, $(proj_files))

TileBasedShadows: $(obj_files)
	@$(CXX) $(obj_files) -L./../../External/AntTweakBar/lib -lGLEW -lAntTweakBar -lSDL2 -lGL -lX11 -o $@
	mv $@ ./../../linoglx64

.PHONY: clean rebuild
clean:
	rm -f *.o ./TileBasedShadows

rebuild:
	$(MAKE) clean
	$(MAKE) TileBasedShadows

ifeq ($(strip $(CXXFLAGS)),)
CXXFLAGS=-O2 -s -DNDEBUG
endif

$(obj_files): %.o: %.cpp ./../includes/stdafx.h ./../includes/unix_port.h
	@echo $<
	@$(CXX) -c $< $(CXXFLAGS) -fpermissive -Wno-unused-result -Wno-maybe-uninitialized \
    -std=c++11 -include ./../includes/unix_port.h -DUNIX_PORT -D__cdecl="" $(incl) -o $@
