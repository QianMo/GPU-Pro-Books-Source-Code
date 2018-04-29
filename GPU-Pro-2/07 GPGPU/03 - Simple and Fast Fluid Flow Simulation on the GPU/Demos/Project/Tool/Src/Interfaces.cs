using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tool
{
    public interface IRealTime
    {
        System.Windows.Forms.ToolStripItem[] GetMenuItems   ();
        System.Windows.Forms.ToolStripItem[] GetLoadItems   ();

        System.Windows.Forms.MouseEventHandler GetMouseClick (); 
        
        void    Create              (IntPtr _WindowHandle, Int32 _Width, Int32 _Height);
        void    UpdateUtilitary     ();
        
        float    UpdateGraphics      (float _dt);
        void    UpdateSimulation    (float _dt);
        void    Release             ();
    }

    public interface IImporter
    {
        void    Create     (System.Windows.Forms.Form _form);
        void    Load       (String _fileName);
    }
}
