#include "Input.h"

namespace NGraphics
{
    CInput::CInput()
    {
        m_KeysDown.reset();
        m_KeysUp.set();
        m_KeysPressed.reset();
        m_KeysReleased.reset();

        m_MouseButtonsDown.reset();
        m_MouseButtonsUp.set();
        m_MouseButtonsPressed.reset();
        m_MouseButtonsReleased.reset();

        m_MousePosition[ 0 ] = 0;
        m_MousePosition[ 1 ] = 0;

        m_DeltaMousePosition[ 0 ] = 0;
        m_DeltaMousePosition[ 1 ] = 0;
    }

    void CInput::Reset()
    {
        m_KeysPressed.reset();
        m_KeysReleased.reset();

        m_MouseButtonsPressed.reset();
        m_MouseButtonsReleased.reset();

        m_DeltaMousePosition[ 0 ] = 0;
        m_DeltaMousePosition[ 1 ] = 0;
    }

    void CInput::ProcessEvent( SDL_Event& event )
    {
        switch ( event.type )
        {
            case SDL_KEYDOWN:
                if ( !event.key.repeat )
                {
                    m_KeysUp[ event.key.keysym.scancode ] = false;
                    m_KeysDown[ event.key.keysym.scancode ] = true;

                    m_KeysPressed[ event.key.keysym.scancode ] = true;
                }
                break;
            case SDL_KEYUP:
                m_KeysDown[ event.key.keysym.scancode ] = false;
                m_KeysUp[ event.key.keysym.scancode ] = true;

                m_KeysReleased[ event.key.keysym.scancode ] = true;
                break;
            case SDL_MOUSEMOTION:
                m_MousePosition[ 0 ] = event.motion.x;
                m_MousePosition[ 1 ] = event.motion.y;
                m_DeltaMousePosition[ 0 ] = event.motion.xrel;
                m_DeltaMousePosition[ 1 ] = event.motion.yrel;
                break;
            case SDL_MOUSEBUTTONDOWN:
                m_MouseButtonsUp[ event.button.button ] = false;
                m_MouseButtonsDown[ event.button.button ] = true;

                m_MouseButtonsPressed[ event.button.button ] = true;
                break;
            case SDL_MOUSEBUTTONUP:
                m_MouseButtonsDown[ event.button.button ] = false;
                m_MouseButtonsUp[ event.button.button ] = true;

                m_MouseButtonsReleased[ event.button.button ] = true;
                break;
        }
    }

    const bool CInput::IsKeyDown( SDL_Scancode key )
    {
        return m_KeysDown[ key ];
    }
    const bool CInput::IsKeyUp( SDL_Scancode key )
    {
        return m_KeysUp[ key ];
    }
    const bool CInput::IsKeyPressed( SDL_Scancode key )
    {
        return m_KeysPressed[ key ];
    }
    const bool CInput::IsKeyReleased( SDL_Scancode key )
    {
        return m_KeysReleased[ key ];
    }

    const bool CInput::IsMouseButtonDown( Uint8 button )
    {
        return m_MouseButtonsDown[ button ];
    }
    const bool CInput::IsMouseButtonUp( Uint8 button )
    {
        return m_MouseButtonsUp[ button ];
    }
    const bool CInput::IsMouseButtonPressed( Uint8 button )
    {
        return m_MouseButtonsPressed[ button ];
    }
    const bool CInput::IsMouseButtonReleased( Uint8 button )
    {
        return m_MouseButtonsReleased[ button ];
    }

    const Sint32* CInput::GetMousePosition()
    {
        return m_MousePosition;
    }
    const Sint32* CInput::GetDeltaMousePosition()
    {
        return m_DeltaMousePosition;
    }
}