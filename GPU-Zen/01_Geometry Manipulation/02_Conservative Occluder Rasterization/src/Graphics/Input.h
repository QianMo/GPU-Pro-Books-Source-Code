#pragma once

#include <SDL.h>
#include <bitset>

namespace NGraphics
{
    class CInput
    {
    private:
        std::bitset<SDL_NUM_SCANCODES> m_KeysDown;
        std::bitset<SDL_NUM_SCANCODES> m_KeysUp;
        std::bitset<SDL_NUM_SCANCODES> m_KeysPressed;
        std::bitset<SDL_NUM_SCANCODES> m_KeysReleased;

        std::bitset<8> m_MouseButtonsDown;
        std::bitset<8> m_MouseButtonsUp;
        std::bitset<8> m_MouseButtonsPressed;
        std::bitset<8> m_MouseButtonsReleased;

        Sint32 m_MousePosition[ 2 ];
        Sint32 m_DeltaMousePosition[ 2 ];

    public:
        CInput();

        void Reset();
        void ProcessEvent( SDL_Event& event );

        const bool IsKeyDown( SDL_Scancode key );
        const bool IsKeyUp( SDL_Scancode key );
        const bool IsKeyPressed( SDL_Scancode key );
        const bool IsKeyReleased( SDL_Scancode key );

        const bool IsMouseButtonDown( Uint8 button );
        const bool IsMouseButtonUp( Uint8 button );
        const bool IsMouseButtonPressed( Uint8 button );
        const bool IsMouseButtonReleased( Uint8 button );

        const Sint32* GetMousePosition();
        const Sint32* GetDeltaMousePosition();
    };
}