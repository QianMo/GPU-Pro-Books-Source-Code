#ifndef browser_h
#define browser_h

#include "mymath/mymath.h"

#include "berkelium/ScriptUtil.hpp"
#include "berkelium/StringUtil.hpp"
#include "berkelium/Berkelium.hpp"
#include "berkelium/Context.hpp"
#include "berkelium/Cursor.hpp"

#include "browser_callbacks_impl.h"

#include <map>
#include <string.h>

enum fileopen_type
{
  OPEN, SAVE, CREATE_FOLDER, SELECT_FOLDER
};

//NOTE: You are responsible for rendering the browser window
class browser_instance
{
  public:
    Berkelium::Window* browser_window;
    char* scroll_buffer;
    bool full_refresh;
    unsigned browser_texture; //type == GLuint
    mm::uvec2 screen;
    bool managed;
};

namespace js
{
  //NOTE: implement this to use the browser GUI
  void bindings_complete( const browser_instance& w ); 
}

class browser : public Berkelium::WindowDelegate
{
  private:
    const std::string default_page;
    std::map< Berkelium::Window*, browser_instance* > instances;
    browser_instance* last_callback_window;

    void set_text( const browser_instance& w );
    void clear( browser_instance& w );

  protected:
    browser() : default_page( "http://www.google.com/" ) {} //singleton
    browser( const browser& );
    browser( browser && );
    browser& operator=( const browser& );

  public:
    void create( browser_instance& w, mm::uvec2 screen, bool managed = false );
    void destroy( browser_instance& w );

    void init( const std::wstring& berkelium_path );
    void update();
    void shutdown(); 

    static browser& get()
    {
      static browser instance;
      return instance;
    }

    virtual void onPaint( Berkelium::Window* wini,
                          const unsigned char* bitmap_in,
                          const Berkelium::Rect& bitmap_rect,
                          size_t num_copy_rects,
                          const Berkelium::Rect* copy_rects,
                          int dx,
                          int dy,
                          const Berkelium::Rect& scroll_rect );
    virtual void onJavascriptCallback( Berkelium::Window* win,
                                        void* reply_msg,
                                        Berkelium::URLString origin,
                                        Berkelium::WideString func_name,
                                        Berkelium::Script::Variant* args,
                                        size_t num_args );
    virtual void onScriptAlert( Berkelium::Window* win,
                                Berkelium::WideString message,
                                Berkelium::WideString defaultValue,
                                Berkelium::URLString url,
                                int flags,
                                bool& success,
                                Berkelium::WideString& value );
    virtual void onConsoleMessage( Berkelium::Window* win,
                                   Berkelium::WideString message,
                                   Berkelium::WideString sourceId,
                                   int line_no );
    virtual void onLoad( Berkelium::Window* win );
    /*virtual void onRunFileChooser( Berkelium::Window* win, //TODO doesn't really seem to work, not being called
                                   int mode, 
                                   Berkelium::WideString title, 
                                   Berkelium::FileString defaultFile );*/
    virtual void onAddressBarChanged( Berkelium::Window* win, 
                                      Berkelium::URLString newURL ){}
    virtual void onStartLoading( Berkelium::Window* win,
                                 Berkelium::URLString newURL ){}
    virtual void onCrashedWorker( Berkelium::Window* win );
    virtual void onCrashedPlugin( Berkelium::Window* win,
                                  Berkelium::WideString pluginName );
    virtual void onProvisionalLoadError( Berkelium::Window* win,
                                         Berkelium::URLString url, 
                                         int errorCode, 
                                         bool isMainFrame );
    /*virtual void onNavigationRequested( Berkelium::Window* win, //TODO doesn't really seem to work, not being called
                                        Berkelium::URLString newURL,
                                        Berkelium::URLString referrer,
                                        bool isNewWindow,
                                        bool& cancelDefaultAction );*/
    virtual void onCursorUpdated( Berkelium::Window* win, 
                                  const Berkelium::Cursor& new_cursor );
    virtual void onLoadingStateChanged( Berkelium::Window* win,
                                        bool isLoading ){}
    virtual void onCrashed( Berkelium::Window* win );
    virtual void onUnresponsive( Berkelium::Window* win );
    virtual void onResponsive( Berkelium::Window* win ){}
    virtual void onCreatedWindow( Berkelium::Window* win,
                                  Berkelium::Window* newWin,
                                  const Berkelium::Rect& initialRect ){}

    //NOTE: implement these to use the browser GUI
    //leave just an empty implementation if unsure
    //not critical functionality
    /*virtual void onResizeRequested( Berkelium::Window* win, //TODO doesnt really work, not being called
                                    int x, 
                                    int y, 
                                    int newWidth, 
                                    int newHeight );*/
    virtual void onTitleChanged( Berkelium::Window* win, 
                                 Berkelium::WideString title );
    

    browser_instance& get_last_callback_window();

    void navigate( browser_instance& w, const std::string& url );

    void resize( browser_instance& w, mm::uvec2 screen );

    void execute_javascript( const browser_instance& w, const std::wstring& str );

    void mouse_moved( const browser_instance& w, mm::vec2 pos )
    {
      w.browser_window->mouseMoved( pos.x * w.screen.x, pos.y * w.screen.y );
    }

    void text_entered( const browser_instance& w, const std::wstring& str )
    {
      w.browser_window->textEvent( str.c_str(), str.length() );
    }

    void mouse_button_event( const browser_instance& w, unsigned button, bool pressed )
    {
      w.browser_window->mouseButton( button, pressed );
    }

    void mouse_wheel_moved( const browser_instance& w, float amount )
    {
      w.browser_window->mouseWheel( 0, amount );
    }

    void select_file( std::vector<std::string>& vec, const std::string& title, bool allow_multiple, fileopen_type type );

  private:
    std::map<std::wstring, fun> the_callbacks;

    void handle_callbacks( const std::wstring& func, Berkelium::Script::Variant* args )
    {
      the_callbacks[func]( args );
    }

  public:

    void register_callback( const std::wstring& function_name, const fun& f )
    {
      the_callbacks[function_name] = f;
    }

    void unregister_callback( const std::wstring& function_name )
    {
      the_callbacks.erase( function_name );
    }
};

#endif
