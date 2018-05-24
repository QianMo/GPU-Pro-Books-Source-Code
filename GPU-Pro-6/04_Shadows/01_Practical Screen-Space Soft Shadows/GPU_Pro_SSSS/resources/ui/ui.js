//startup

$(function()
{
	//for in-browser dev
  //init_js();
}
);

var blur_res_radio = false;
var penumbra_res_radio = true;
var supersampling_radio = true;
var exponential_radio = false;

function init_js()
{ 
  $('button').button();
	$(document).tooltip();
 
  $('#about_dialog').dialog({ autoOpen: false, show: { effect: 'drop', direction: 'up', duration: 500 }, width: 450 });
  $('#options_dialog').dialog({ autoOpen: false, show: { effect: 'drop', direction: 'up', duration: 500 } });
  $('#light_properties_dialog').dialog({ autoOpen: false, show: { effect: 'drop', direction: 'up', duration: 500 } });
  $('#colorpicker_dialog').dialog({ autoOpen: false, show: { effect: 'drop', direction: 'up', duration: 500 } });
 
  $('#about_button').click(function()
  {
    $('#about_dialog').dialog("open");
  });
  
  $('#options_button').click(function()
  {
    $('#options_dialog').dialog("open");
  });
  
  $('#light_properties_button').click(function()
  {
    $('#light_properties_dialog').dialog("open");
  });
  
  $('#fixed_position0').click(function(){ go_to_fixed_position(0); });
  $('#fixed_position1').click(function(){ go_to_fixed_position(1); });
  $('#fixed_position2').click(function(){ go_to_fixed_position(2); });
  $('#fixed_position3').click(function(){ go_to_fixed_position(3); });
  $('#fixed_position4').click(function(){ go_to_fixed_position(4); });
  $('#fixed_position5').click(function(){ go_to_fixed_position(5); });
  $('#fixed_position6').click(function(){ go_to_fixed_position(6); });
  $('#fixed_position7').click(function(){ go_to_fixed_position(7); });
  $('#fixed_position8').click(function(){ go_to_fixed_position(8); });
  
  
  $('#technique_selector').selectmenu({
    change: function( event, data )
    {
      console.log( data.item.label );
      
      if( data.item.label != "SSSS minfilter" &&
          data.item.label != "SSSS blocker search" )
      {
        $('#blur_radio').buttonset('disable');
        $('#penumbra_radio').buttonset('disable');
        $('#supersampling_radio').buttonset('disable');
      }
      else
      {
        $('#blur_radio').buttonset('enable');
        $('#penumbra_radio').buttonset('enable');
        $('#supersampling_radio').buttonset('enable');
      }
      
      if( data.item.label == "Lighting only" ||
          data.item.label == "SSSS minfilter" ||
          data.item.label == "SSSS blocker search" )
      {
        exponential_radio = false;
        $('#exponential_radio_off').prop( "checked", true );
        $('#exponential_radio_on').prop( "checked", false );
        $('#exponential_radio_off').button('refresh');
        $('#exponential_radio_on').button('refresh');
        set_exponential( exponential_radio );
        $('#exponential_radio').buttonset('disable');
      }
      else
      {
        $('#exponential_radio').buttonset('enable');
      }
      
      set_technique( data.item.label );
    }
  });
  
  $('#blur_radio').buttonset();
  $('#penumbra_radio').buttonset();
  $('#supersampling_radio').buttonset();
  $('#exponential_radio').buttonset();
  
  $('#exponential_radio').buttonset('disable');
  
  $('#blur_radio').change(function()
  {
    blur_res_radio = !blur_res_radio;
    
    set_gauss_blur_res( blur_res_radio );
  });
  
  $('#penumbra_radio').change(function()
  {
    penumbra_res_radio = !penumbra_res_radio;
    
    set_penumbra_res( penumbra_res_radio );
  });
  
  $('#supersampling_radio').change(function()
  {
    supersampling_radio = !supersampling_radio;
    
    set_supersampling( supersampling_radio );
  });
  
  $('#exponential_radio').change(function()
  {
    exponential_radio = !exponential_radio;
    
    set_exponential( exponential_radio );
  });
  
  $('#hideui').click(function()
  {
    $('#content').fadeOut(250, function()
    {
      $('#showui').fadeIn(250);
    });
  });
  
  $('#showui').click(function()
  {
    $('#showui').fadeOut(250, function()
    {
      $('#content').fadeIn(250);
    });
  });
  
  $('#add_light').click(function()
  {
    add_light();
  });
 
  $('#remove_light').click(function()
  {
    remove_light();
  });
 
  $.farbtastic('#colorpicker',
  function()
  {
    var hsl_color = $.farbtastic('#colorpicker').hsl;
    var rgb_color = $.farbtastic('#colorpicker').HSLToRGB(hsl_color);
    set_light_color( rgb_color[0].toString() + " " +
                     rgb_color[1].toString() + " " +
                     rgb_color[2].toString() );
    $('#light_color').val($.farbtastic('#colorpicker').color);
    $('#light_color').css('background', $('#light_color').val());
    $('#light_color').css('color', "rgba(" +
                                   ((1.0-rgb_color[0])*255).toString() + ", " +
                                   ((1.0-rgb_color[1])*255).toString() + ", " +
                                   ((1.0-rgb_color[2])*255).toString() + ", 1.0)" );
  });

  $('#light_color').click(
  function()
  {
    $('#colorpicker_dialog').dialog('open');
  });
  //$('#colorpicker').draggable();
  set_light_color_cpp(255, 255, 255);
  
  $('#light_radius_slider').slider({
  min: 1, max: 100, value: 30, slide: function(event, ui)
  {
    $('#light_radius_slider_value').val(ui.value);
    set_light_radius(ui.value);
  } });
  $('#light_radius_slider_value').val( 30 );
  
  $('#light_size_slider').slider({
  min: 1, max: 1000, value: 150, slide: function(event, ui)
  {
    $('#light_size_slider_value').val(ui.value / 1000);
    set_light_size(ui.value / 1000);
  } });
  $('#light_size_slider_value').val( 0.15 );

  $('#set_resolution').click(function(){ set_resolution( $('#resolution_selector').val() ); });
 
  $('#reload_shaders_button').click(function()
  {
    reload_shaders();
  });
 
  //var duration = 250; 
  var duration = 3000;
  $('#intro').fadeIn(duration, function()
  { 
    $('#intro').fadeOut(duration, function()
    {
      $(document.body).animate({ "background-color": "rgba(0, 0, 0, 0)" }, duration);
      $('#content').fadeIn(duration);
    }); 
  });
}