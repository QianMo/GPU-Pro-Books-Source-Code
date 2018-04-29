using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;
using System.IO;

using System.Threading;

namespace Tool
{

    public class PerformanceMeasure
    {
        private long tBefore, tAfter, cFrequency = 0;       

        [DllImport("Kernel32.dll")]
        private static extern bool QueryPerformanceCounter(out long lpPerformanceCount);

        [DllImport("Kernel32.dll")]
        private static extern bool QueryPerformanceFrequency(out long lpFrequency);

        public PerformanceMeasure() { Begin(); }
        public void Begin()
        {
            QueryPerformanceCounter(out tBefore);
        }

        public Double End()
        {
            QueryPerformanceCounter(out tAfter);
            QueryPerformanceFrequency(out cFrequency);
            return (Double)(tAfter - tBefore) / (Double)cFrequency;
        }
    }

	public class Timer
	{
        private IRealTime                   m_pApp;
        Double                              m_frameRate;
        System.Windows.Forms.NumericUpDown  m_SharedNumUD;    

        public Timer(IRealTime _ToUpdate) { m_pApp = _ToUpdate; }

        delegate void SetTextCallback(decimal _value);

        /// Thread Safe Set.   
        private void SetFrameRate(decimal _value)
        {
            // InvokeRequired required compares the thread ID of the
            // calling thread to the thread ID of the creating thread.
            // If these threads are different, it returns true.
            if (m_SharedNumUD.InvokeRequired)
            {
                SetTextCallback d = new SetTextCallback(SetFrameRate);
                m_SharedNumUD.Invoke(d, new object[] { _value });
            }
            else
            {
                m_SharedNumUD.Value = _value;
            }
        }

        /// <summary>
        /// <
        /// </summary>
        /// <returns></returns>
        public void SetNumUD(System.Windows.Forms.NumericUpDown _NumUD)
        {
            m_SharedNumUD = _NumUD;
        }

        public void Release     ()  { m_pApp.Release(); }
        
        private static void AddText(FileStream fs, string value)
        {
            byte[] info = new UTF8Encoding(true).GetBytes(value);
            fs.Write(info, 0, info.Length);
        }

		public void Run()
		{
            Double dt = 1.0f / 30.0f;
            PerformanceMeasure pmHole = new PerformanceMeasure();
            List<Double> AverageFR = new List<Double>();
            
            try
            {
                
                while (true)
                {
                    m_pApp.UpdateUtilitary();

                    {
                        pmHole.Begin();                        
                        m_frameRate  = m_pApp.UpdateGraphics((float)dt);  
                        m_pApp.UpdateSimulation((float)dt / 2.0f);
                        m_pApp.UpdateSimulation((float)dt / 2.0f);
                        Double cpuFrameRate= pmHole.End();
                        /*
                        if (m_frameRate == 0)
                            m_frameRate = cpuFrameRate;
                        */
                        //Thread.Sleep(10);
       
                        Double fr = 0.0f;
                        if (m_frameRate > 0.0f)
                            fr = (1.0f / m_frameRate);

                        AverageFR.Add(fr);
                        if (AverageFR.Count > 1)
                        {
                            AverageFR.RemoveAt(0);
                        }
                        Double average = 0;
                        for (Int32 i = 0; i < AverageFR.Count; ++i)
                            average += AverageFR[i];

                        average = average / (Double)AverageFR.Count;

                        SetFrameRate((decimal)average);
                    }
                }/// While(true)
                 /// 
                
            }
            catch (ThreadAbortException ab)
            {
                String ms = ab.Message;
                m_pApp.Release();
                ///< Cancel Abort with : ResetAbort 
            }

            /*
            Double GraphicsFrameMeasure             = 0;
            Double PhysicsFrameMeasure              = 0;
            Double ComputationalFrameTime           = 0;           
            
            FileStream fStream = File.Open(@"../../Version/physstep.txt", FileMode.OpenOrCreate);

            PerformanceMeasure pmHole       = new PerformanceMeasure();
            PerformanceMeasure pmGraphics   = new PerformanceMeasure();
            PerformanceMeasure pmSimu       = new PerformanceMeasure(); 
            while (true)
			{
                m_pApp.UpdateUtilitary();
                ComputationalFrameTime = dt;
                {
                    pmHole.Begin();
                    {
                        ///< Graphics
                        {
                            pmGraphics.Begin();
                            m_pApp.UpdateGraphics((float)dt);
                            GraphicsFrameMeasure = pmGraphics.End();
                            //AddText(fStream, "Graphics Computational Step " + (GraphicsFrameMeasure * 1000.0f).ToString() + "\r\n");
                            ComputationalFrameTime -= GraphicsFrameMeasure;
                        }

                        PhysicsFrameMeasure = 0;
                        int SimuStepsCounts = 0;
                       // while (ComputationalFrameTime - PhysicsFrameMeasure > 0 && SimuStepsCounts < 2)
                        {
                            pmSimu.Begin();
                            m_pApp.UpdateSimulation((float)dt / 2.0f);
                            PhysicsFrameMeasure = pmSimu.End();
                            ComputationalFrameTime -= PhysicsFrameMeasure;
                            SimuStepsCounts++;
                        }                

                        AddText(fStream, "Computational Frame Time " + (ComputationalFrameTime * 1000.0f).ToString() + "\r\n");

                    } 

                                             
                    }

                    int millisecondsToWait = (int)(ComputationalFrameTime * 1000.0f);
                    if (millisecondsToWait>0 && millisecondsToWait < dt*1000.0f)
                        Thread.Sleep(millisecondsToWait);

                    m_frameRate = pmHole.End();
                    SetFrameRate((decimal)Math.Min((1.0f / m_frameRate), 300));
                }
             */   
			}
             
		}
            

        



        
	}

