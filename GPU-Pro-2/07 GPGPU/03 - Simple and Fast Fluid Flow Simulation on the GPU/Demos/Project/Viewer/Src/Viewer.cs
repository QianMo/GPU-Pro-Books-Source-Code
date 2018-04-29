using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace V
{
    public class Viewer : Tool.IRealTime 
    {
        IntPtr          m_WindowHandle;
        M.MRenderer     m_Renderer;

        System.Windows.Forms.ToolStripMenuItem[] m_ControlMenu;
        System.Windows.Forms.ToolStripMenuItem[] m_LoadMenu;

        public System.Windows.Forms.ToolStripItem[] GetMenuItems()  { return m_ControlMenu; }
        public System.Windows.Forms.ToolStripItem[] GetLoadItems()  { return m_LoadMenu;  }

        public System.Windows.Forms.MouseEventHandler GetMouseClick() { return new System.Windows.Forms.MouseEventHandler(this.MouseClickEvent); }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="_WindowHandle"></param>
        /// <param name="_Width"></param>
        /// <param name="_Height"></param>
        public void Create(IntPtr _WindowHandle, Int32 _Width, Int32 _Height)
        {
            m_WindowHandle  = _WindowHandle;
            m_Renderer      = new M.MRenderer(m_WindowHandle, _Width, _Height);
            m_Renderer.Load("DX10TDGPUSolver");

            CreateControlMenuItems();
            CreateLoadMenuItems();
        }
        /// <summary>
        /// <
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        public void MouseClickEvent(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            m_Renderer.MouseClick();
        }

        /// <summary>
        /// /<
        /// </summary>
        private void CreateLoadMenuItems()
        {
            m_LoadMenu                      = new System.Windows.Forms.ToolStripMenuItem[2];       

            ///<
            m_LoadMenu[0] = new System.Windows.Forms.ToolStripMenuItem();
            m_LoadMenu[0].CheckOnClick = true;
            m_LoadMenu[0].Name = "DX10GPUSolver";
            m_LoadMenu[0].ShowShortcutKeys = false;
            m_LoadMenu[0].Size = new System.Drawing.Size(111, 22);
            m_LoadMenu[0].Text = "Load DX10 GPU Solver";
            m_LoadMenu[0].Click += new System.EventHandler(this.LoadDX10GPUSolver);

            ///<
            m_LoadMenu[1] = new System.Windows.Forms.ToolStripMenuItem();
            m_LoadMenu[1].CheckOnClick = true;
            m_LoadMenu[1].Name = "DX10TDGPUSolver";
            m_LoadMenu[1].ShowShortcutKeys = false;
            m_LoadMenu[1].Size = new System.Drawing.Size(111, 22);
            m_LoadMenu[1].Text = "Load DX10 3D GPU Solver";
            m_LoadMenu[1].Click += new System.EventHandler(this.LoadDX10TDGPUSolver);

        }

        private void CreateControlMenuItems()
        {
            m_ControlMenu = new System.Windows.Forms.ToolStripMenuItem[1];

            m_ControlMenu[0] = new System.Windows.Forms.ToolStripMenuItem();
            m_ControlMenu[0].CheckOnClick = true;
            m_ControlMenu[0].Name = "Pause";
            m_ControlMenu[0].ShowShortcutKeys = false;
            m_ControlMenu[0].Size = new System.Drawing.Size(111, 22);
            m_ControlMenu[0].Text = "Pause Simulation";
            m_ControlMenu[0].Click += new System.EventHandler(this.PauseSimulation);


        }

        /// <summary>
        /// Load Menu Items.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        ///

        private void LoadDX10GPUSolver(object sender, EventArgs e)
        {
            m_Renderer.Load("DX10GPUSolver");
        }
        private void LoadDX10TDGPUSolver(object sender, EventArgs e)
        {
            m_Renderer.Load("DX10TDGPUSolver");
        }
        

        /// <summary>
        /// Control Menu Items
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void PauseSimulation(object sender, EventArgs e)
        {
            m_Renderer.UpdateOnOff();
        }

        



        /// <summary>
        /// < Return GPU fps.
        /// </summary>
        /// <param name="_dt"></param>
        public float UpdateGraphics(float _dt)
        {
            return m_Renderer.UpdateGraphics(_dt);
        }

        public void UpdateSimulation(float _dt)
        {
            m_Renderer.UpdateSimulation(_dt);
        }

        public void UpdateUtilitary()
        {
            m_Renderer.UpdateUtilitary();
        }

        public void Release()
        {
            m_Renderer.Release();
        }       
    }
}
