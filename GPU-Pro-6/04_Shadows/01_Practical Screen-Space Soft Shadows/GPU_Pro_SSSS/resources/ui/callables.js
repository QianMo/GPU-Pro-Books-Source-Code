//cpp to js calls

//$('#mirror_x').prop('checked', mirror_x);
//$('#intensity_slider').slider('value', intensity * 100);

function bindings_complete()
{
  init_js();
}

function cpp_to_js(str)
{
  
}

function set_resolutions_autocomplete( the_source )
{
  $('#resolution_selector').autocomplete({source: the_source});
}

function set_user_pos( x, y, z )
{
  $('#user_cam_pos').val( x.toString() + ", " + 
                          y.toString() + ", " + 
                          z.toString() );
}

function set_user_view( x, y, z )
{
  $('#user_cam_view').val( x.toString() + ", " + 
                           y.toString() + ", " + 
                           z.toString() );
}

function set_user_up( x, y, z )
{
  $('#user_cam_up').val( x.toString() + ", " + 
                         y.toString() + ", " + 
                         z.toString() );
}

function set_light_pos( x, y, z )
{
  $('#light_cam_pos').val( x.toString() + ", " + 
                          y.toString() + ", " + 
                          z.toString() );
}

function set_light_view( x, y, z )
{
  $('#light_cam_view').val( x.toString() + ", " + 
                           y.toString() + ", " + 
                           z.toString() );
}

function set_light_up( x, y, z )
{
  $('#light_cam_up').val( x.toString() + ", " + 
                         y.toString() + ", " + 
                         z.toString() );
}

function set_light_radius_cpp( radius )
{
  $('#light_radius_slider_value').val( radius );
  $('#light_radius_slider').slider( 'value', radius );
}

function set_light_size_cpp( size )
{
  $('#light_size_slider_value').val( size );
  $('#light_size_slider').slider( 'value', size*1000 );
}

function set_light_color_cpp(r, g, b)
{
  var a = [r, g, b];
  var b = a.map(
  function(x)
  {                                     //For each array element
    x = parseInt(x).toString(16);       //Convert to a base16 string
    return (x.length==1) ? "0"+x : x; //Add zero if we get only one character
  });
   
  b = "#"+b.join("");
   
  $.farbtastic('#colorpicker').setColor(b);
}