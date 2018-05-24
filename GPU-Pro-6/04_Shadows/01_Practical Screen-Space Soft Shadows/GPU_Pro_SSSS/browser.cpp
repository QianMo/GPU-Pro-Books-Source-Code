#include "browser.h"
#include <berkelium/Widget.hpp>

#include <sstream>
#include <vector>
#include <string>
#include <locale>
#include <codecvt>

#include <GL/glew.h>
#include <SFML/System.hpp>

#define BROWSER_USE_EXCEPTIONS

void browser::select_file( std::vector<std::string>& vec, const std::string& title, bool allow_multiple, fileopen_type type )
{
#ifdef __unix__
  GtkFileChooserAction fca = GTK_FILE_CHOOSER_ACTION_OPEN;

  switch( type )
  {
    case OPEN:
      {
        fca = GTK_FILE_CHOOSER_ACTION_OPEN;
        break;
      }
    case SAVE:
      {
        fca = GTK_FILE_CHOOSER_ACTION_SAVE;
        break;
      }
    case CREATE_FOLDER:
      {
        fca = GTK_FILE_CHOOSER_ACTION_CREATE_FOLDER;
        break;
      }
    case SELECT_FOLDER:
      {
        fca = GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER;
        break;
      }
  }

  //display file dialog, load texture
  GtkWidget* dialog = gtk_file_chooser_dialog_new( title.c_str(),
                      ( GtkWindow* )the_bg_window,
                      fca,
                      GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
                      GTK_STOCK_OPEN, GTK_RESPONSE_ACCEPT,
                      NULL );
  gtk_file_chooser_set_select_multiple( GTK_FILE_CHOOSER( dialog ), allow_multiple );

  if( gtk_dialog_run( GTK_DIALOG( dialog ) ) == GTK_RESPONSE_ACCEPT )
  {
    GSList* filenames = gtk_file_chooser_get_filenames( GTK_FILE_CHOOSER( dialog ) );

    while( filenames )
    {
      vec.push_back( std::string( ( char* )filenames->data ) );
      g_free( filenames->data );
      filenames = filenames->next;
    }

    g_slist_free( filenames );
  }

  gtk_widget_destroy( dialog );
#endif

#ifdef _WIN32
  OPENFILENAME ofn;
  char filename[MAX_PATH * 100] = "";

  ZeroMemory( &ofn, sizeof( ofn ) );

  ofn.lStructSize = sizeof( ofn ); // SEE NOTE BELOW
  ofn.hwndOwner = 0;
  ofn.lpstrFilter = "All Files (*.*)\0*.*\0";
  ofn.lpstrFile = filename;
  ofn.nMaxFile = MAX_PATH;
  ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | ( allow_multiple ? OFN_ALLOWMULTISELECT : 0 );
  ofn.lpstrDefExt = "";

  if( GetOpenFileName( &ofn ) )
  {
    int counter = 0;
    vec.push_back( std::string() );

    for( int c = 0; c < MAX_PATH * 100 - 1; ++c )
    {

      if( filename[c] != '\0' )
      {
        vec[counter].push_back( filename[c] );
      }
      else
      {
        if( filename[c + 1] == '\0' )
        {
          break;
        }

        ++counter;
        vec.push_back( std::string() );
      }
    }

    if( vec.size() > 1 )
    {
      //user selected more files
      vec[0] += '/';

      for( unsigned int c = 1; c < vec.size(); ++c )
      {
        vec[c].insert( vec[c].begin(), vec[0].begin(), vec[0].end() );
      }

      vec.erase( vec.begin() );
    }
  }

#endif

  std::for_each( vec.begin(), vec.end(),
                 [&]( std::string & s )
  {
    std::replace( s.begin(), s.end(), '\\', '/' );
  }
               );
}

void browser::onCursorUpdated( Berkelium::Window* win, 
                               const Berkelium::Cursor& new_cursor )
{
  SetCursor( new_cursor.GetCursor() );
}

/*void browser::onRunFileChooser( Berkelium::Window* win, 
                                int mode, 
                                Berkelium::WideString title, 
                                Berkelium::FileString defaultFile )
{
  std::cout << "run file chooser" << std::endl;

  std::vector< std::string > filenames;
  std::wstring titlestd( title.mData, title.mLength );
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
  std::string title_normal = conv.to_bytes( titlestd );
  switch( mode )
  {
    case Berkelium::FileChooserType::FileOpen:
    {
      select_file( filenames, title_normal != "" ? title_normal : "Open", false, fileopen_type::OPEN );
      break;
    }
    case Berkelium::FileChooserType::FileOpenFolder:
    {
      select_file( filenames, title_normal != "" ? title_normal : "Select folder", false, fileopen_type::SELECT_FOLDER );
      break;
    }
    case Berkelium::FileChooserType::FileOpenMultiple:
    {
      select_file( filenames, title_normal != "" ? title_normal : "Open multiple files", true, fileopen_type::OPEN );
      break;
    }
    case Berkelium::FileChooserType::FileSaveAs:
    {
      select_file( filenames, title_normal != "" ? title_normal : "Save as", true, fileopen_type::SAVE );
      break;
    }
    default:
      break;
  }

  //TODO the api doesn't say where to save the return value...
  //so we'll just call a js function...
#ifdef BROWSER_USE_EXCEPTIONS
  try
#endif
  {
    browser_instance& w = *instances.at( win );

    for( auto& s : filenames )
    {
      std::wstringstream ss;
      ss << L"filechooser_callback('" << conv.from_bytes(s) << L"');";
      execute_javascript( w, ss.str() );
    }
  }
#ifdef BROWSER_USE_EXCEPTIONS
  catch(...){}
#endif
}*/

void browser::onCrashedWorker( Berkelium::Window* win )
{
  std::cerr << "A berkelium worker has just crashed!" << std::endl;
}

void browser::onCrashedPlugin( Berkelium::Window* win,
                               Berkelium::WideString pluginName )
{
  std::wcerr << L"The berkelium plugin: " << std::wstring(pluginName.mData, pluginName.mLength) << " has just crashed!" << std::endl;
}

void browser::onProvisionalLoadError( Berkelium::Window* win,
                                      Berkelium::URLString url, 
                                      int errorCode, 
                                      bool isMainFrame )
{
  if( isMainFrame )
  {
    std::cerr << "Failed to load main frame (window): " << std::string(url.mData, url.mLength) << std::endl;
  }
  else
  {
    std::cerr << "Failed to load XHR or iframe: " << std::string(url.mData, url.mLength) << std::endl;
  }

  std::cerr << "Error code: " << errorCode << std::endl;
}

/*void browser::onNavigationRequested( Berkelium::Window* win,
                            Berkelium::URLString newURL,
                            Berkelium::URLString referrer,
                            bool isNewWindow,
                            bool& cancelDefaultAction )
{
  cancelDefaultAction = false;

#ifdef BROWSER_USE_EXCEPTIONS
  try
#endif
  {
    browser_instance& w = *instances.at( win );

    if( isNewWindow )
    {
      std::cout << "Navigation requested by: " << std::string(referrer.mData, referrer.mLength) << std::endl;
      std::cout << "To: " << std::string(newURL.mData, newURL.mLength) << std::endl;

      browser_instance& nw = *new browser_instance();
      create( nw, w.screen, true ); //non-user-created window --> managed
    }
  }
#ifdef BROWSER_USE_EXCEPTIONS
  catch(...){}
#endif
}*/

void browser::onCrashed( Berkelium::Window* win )
{
  std::cerr << "A berkelium window just crashed!" << std::endl;
}

void browser::onUnresponsive( Berkelium::Window* win )
{
  std::cerr << "A berkelium window is unresponsive!" << std::endl;
}

void browser::update()
{
  Berkelium::update();
}

void browser::navigate( browser_instance& w, const std::string& url )
{
  std::cout << "Browser goto: " << url << std::endl;
  clear( w );
  w.browser_window->navigateTo( url.data(), url.length() );
}

void browser::destroy( browser_instance& w )
{
  glDeleteTextures( 1, &w.browser_texture );
  delete [] w.scroll_buffer;
  delete w.browser_window;
  instances.erase( w.browser_window );
}

void browser::init( const std::wstring& berkelium_path )
{
#ifdef _WIN32 //on win32 use the new initialization method (only works with the new berkelium 11)
  std::wstringstream path;
  path << berkelium_path;
  Berkelium::init( Berkelium::FileString::empty(), Berkelium::FileString::point_to( path.str().data(), path.str().length() ) );
#else
  Berkelium::init( Berkelium::FileString::empty() );
#endif
}

void browser::create( browser_instance& w, mm::uvec2 screen, bool managed )
{
  w.managed = managed;

  glGenTextures( 1, &w.browser_texture );
  glBindTexture( GL_TEXTURE_2D, w.browser_texture );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );

  w.screen = screen;
  w.scroll_buffer = new char[screen.x * ( screen.y + 1 ) * 4];

  Berkelium::Context* browser_context = 0;

#ifdef BROWSER_USE_EXCEPTIONS
  try
#endif
  {
    Berkelium::Context* browser_context = Berkelium::Context::create();
    w.browser_window = Berkelium::Window::create( browser_context ); //possibly this creates the texture...
  }
#ifdef BROWSER_USE_EXCEPTIONS
  catch( std::exception e )
  {
    std::cerr << "Error creating Berkelium context: " << e.what() << std::endl;
    return;
  }
#endif

  delete browser_context;

  instances[w.browser_window] = &w;

  w.browser_window->setDelegate( this );
  w.browser_window->resize( screen.x, screen.y );
  w.browser_window->setTransparent( true );
  w.browser_window->focus();

  navigate( w, default_page );
}

void browser::resize( browser_instance& w, mm::uvec2 screen )
{
  delete [] w.scroll_buffer;
  w.scroll_buffer = new char[screen.x * ( screen.y + 1 ) * 4];
  w.browser_window->resize( screen.x, screen.y );
  w.full_refresh = true;
  w.screen = screen;
}

void browser::shutdown()
{
  //destroy managed (non-user created) windows
  for( auto it = instances.begin(); it != instances.end(); )
  {
    browser_instance& w = *it->second;
    
    if( w.managed )
      destroy( w );
    else
      instances.erase( it );

    it = instances.begin();
  }

  Berkelium::destroy();
}

void browser::clear( browser_instance& w )
{
  unsigned char black = 0;
  glBindTexture( GL_TEXTURE_2D, w.browser_texture );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_R8, 1, 1, 0, GL_RED, GL_UNSIGNED_BYTE, &black );
  w.full_refresh = true;
}

void browser::onPaint( Berkelium::Window* wini,
                        const unsigned char* bitmap_in,
                        const Berkelium::Rect& bitmap_rect,
                        size_t num_copy_rects,
                        const Berkelium::Rect* copy_rects,
                        int dx,
                        int dy,
                        const Berkelium::Rect& scroll_rect )
{
  const int bytes_per_pixel = 4;

#ifdef BROWSER_USE_EXCEPTIONS
  try
#endif
  {
    browser_instance& w = *instances.at( wini );
    
    glBindTexture( GL_TEXTURE_2D, w.browser_texture );

    if( w.full_refresh )
    {
      if( bitmap_rect.left() != 0 ||
          bitmap_rect.top() != 0 ||
          bitmap_rect.right() != w.screen.x ||
          bitmap_rect.bottom() != w.screen.y )
      {
        return;
      }
      else
      {
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, w.screen.x, w.screen.y, 0, GL_BGRA, GL_UNSIGNED_BYTE, ( void* )bitmap_in );
        w.full_refresh = false;
        return;
      }
    }

    if( dx != 0 || dy != 0 )
    {
      Berkelium::Rect scrolled_rect = scroll_rect.translate( -dx, -dy );
      Berkelium::Rect scrolled_shared_rect = scroll_rect.intersect( scrolled_rect );

      if( scrolled_shared_rect.width() > 0 && scrolled_shared_rect.height() > 0 )
      {
        Berkelium::Rect shared_rect = scrolled_shared_rect.translate( dx, dy );

        int wid = scrolled_shared_rect.width();
        int hig = scrolled_shared_rect.height();

        int inc = 1;
        char* output_buffer = w.scroll_buffer;
        char* input_buffer = w.scroll_buffer + ( w.screen.x * 1 * bytes_per_pixel );

        int jj = 0;

        if( dy > 0 )
        {
          output_buffer = w.scroll_buffer + ( ( scrolled_shared_rect.top() + hig + 1 ) * w.screen.x - hig * wid ) * bytes_per_pixel;
          input_buffer = w.scroll_buffer;
          inc = -1;
          jj = hig - 1;
        }

        glGetTexImage( GL_TEXTURE_2D, 0, GL_BGRA, GL_UNSIGNED_BYTE, input_buffer );

        for( ; jj < hig && jj >= 0; jj += inc )
        {
          memcpy( output_buffer + ( jj * wid ) * bytes_per_pixel,
                  input_buffer + ( ( scrolled_shared_rect.top() + jj ) * w.screen.x + scrolled_shared_rect.left() ) * bytes_per_pixel,
                  wid * bytes_per_pixel );
        }

        glTexSubImage2D( GL_TEXTURE_2D, 0, shared_rect.left(), shared_rect.top(), shared_rect.width(), shared_rect.height(), GL_BGRA, GL_UNSIGNED_BYTE, output_buffer );
      }
    }

    for( size_t i = 0; i < num_copy_rects; i++ )
    {
      int wid = copy_rects[i].width();
      int hig = copy_rects[i].height();
      int top = copy_rects[i].top() - bitmap_rect.top();
      int left = copy_rects[i].left() - bitmap_rect.left();

      for( int jj = 0; jj < hig; jj++ )
      {
        memcpy( w.scroll_buffer + jj * wid * bytes_per_pixel,
                bitmap_in + ( left + ( jj + top ) * bitmap_rect.width() ) * bytes_per_pixel,
                wid * bytes_per_pixel );
      }

      glTexSubImage2D( GL_TEXTURE_2D, 0, copy_rects[i].left(), copy_rects[i].top(), wid, hig, GL_BGRA, GL_UNSIGNED_BYTE, w.scroll_buffer );
    }

    w.full_refresh = false;
  }
#ifdef BROWSER_USE_EXCEPTIONS
  catch(...){}
#endif
}

template< class c, class s >
class berkelium_string : public Berkelium::WeakString<c>
{
  public:
    berkelium_string( const s& str )
    {
      Berkelium::WeakString<c>::mData = str.data();
      Berkelium::WeakString<c>::mLength = str.length();
    }
};

void browser::set_text( const browser_instance& w ) //cpp to javascript function calling
{
  std::for_each( the_callbacks.begin(), the_callbacks.end(),
                  [&]( std::pair<std::wstring, fun> p )
  {
    Berkelium::WeakString<wchar_t> str;
    str = berkelium_string<wchar_t, std::wstring>( p.first );
    w.browser_window->bind( str, Berkelium::Script::Variant::bindFunction( str, false ) );
  }
                );
}

void browser::onJavascriptCallback( Berkelium::Window* win,
                                    void* reply_msg,
                                    Berkelium::URLString origin,
                                    Berkelium::WideString func_name,
                                    Berkelium::Script::Variant* args,
                                    size_t num_args )
{
  sf::Mutex mtx;
  mtx.lock();
  {
#ifdef BROWSER_USE_EXCEPTIONS
    try
#endif
    {
      last_callback_window = instances.at(win);
    }
#ifdef BROWSER_USE_EXCEPTIONS
    catch(...){}
#endif

    std::wstring func( func_name.data(), func_name.length() );

    if( num_args > 0 )
    {
      handle_callbacks( func, args );
    }
    else
      handle_callbacks( func, 0 );
  }
  mtx.unlock();
}

void browser::onScriptAlert( Berkelium::Window* win,
                              Berkelium::WideString message,
                              Berkelium::WideString defaultValue,
                              Berkelium::URLString url,
                              int flags,
                              bool& success,
                              Berkelium::WideString& value )
{
  std::wstring content( message.data(), message.length() );
  std::wcout << content << std::endl;
}

void browser::onConsoleMessage( Berkelium::Window* win,
                                Berkelium::WideString message,
                                Berkelium::WideString sourceId,
                                int line_no )
{
  std::wstring content( message.data(), message.length() );
  std::wcout << L"Line " << line_no << ": " << content << std::endl;
}

browser_instance& browser::get_last_callback_window()
{
  return *last_callback_window;
}

void browser::onLoad( Berkelium::Window* win )
{
#ifdef BROWSER_USE_EXCEPTIONS
  try
#endif
  {
  browser_instance& w = *instances.at(win);

  sf::Mutex mtx;
  mtx.lock();
  {
    std::cout << "Browser bindings set." << std::endl;
    set_text( w );
    js::bindings_complete( w );
  }
  mtx.unlock();
  }
#ifdef BROWSER_USE_EXCEPTIONS
  catch(...){}
#endif
}

void browser::execute_javascript( const browser_instance& w, const std::wstring& str )
{
  Berkelium::WeakString<wchar_t> s;
  s = berkelium_string<wchar_t, std::wstring>( str );
  w.browser_window->executeJavascript( s );
}