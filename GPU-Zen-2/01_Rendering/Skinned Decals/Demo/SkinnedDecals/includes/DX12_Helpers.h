#ifndef DX12_HELPERS_H
#define DX12_HELPERS_H

class ScopedTimerQuery
{
public:
  ScopedTimerQuery(DX12_CmdList *cmdList, gpuCmdOrders order, DX12_TimerQuery *timerQuery)
  {
    this->cmdList = cmdList;
    this->order = order;
    this->timerQuery = timerQuery;

    BeginTimerQueryCmd cmd;
    cmd.timerQuery = timerQuery;
    cmdList->AddGpuCmd(cmd, order);
  }

  ~ScopedTimerQuery()
  {
    EndTimerQueryCmd cmd;
    cmd.timerQuery = timerQuery;
    cmdList->AddGpuCmd(cmd, order);
  }

private:
  DX12_CmdList *cmdList;
  gpuCmdOrders order;
  DX12_TimerQuery *timerQuery;

};

class ScopedGpuMarker
{
public:
  ScopedGpuMarker(DX12_CmdList *cmdList, gpuCmdOrders order, const char *name)
  {
    this->cmdList = cmdList;
    this->order = order;

    BeginGpuMarkerCmd cmd;
    cmd.SetMarkerName(name);
    cmdList->AddGpuCmd(cmd, order);
  }

  ~ScopedGpuMarker()
  {
    EndGpuMarkerCmd cmd;
    cmdList->AddGpuCmd(cmd, order);
  }

private:
  DX12_CmdList *cmdList;
  gpuCmdOrders order;

};

#define SCOPED_TIMER_QUERY(cmdList_, order_, timerQuery_) ScopedTimerQuery scopedTimerQuery(cmdList_, order_, timerQuery_);

#define BEGIN_TIMER_QUERY(cmdList_, order_, timerQuery_) \
  { \
    BeginTimerQueryCmd cmd; \
    cmd.timerQuery = timerQuery_; \
    cmdList_->AddGpuCmd(cmd, order_); \
  }

#define END_TIMER_QUERY(cmdList_, order_, timerQuery_) \
  { \
    EndTimerQueryCmd cmd; \
    cmd.timerQuery = timerQuery_; \
    cmdList_->AddGpuCmd(cmd, order_); \
  }

#ifdef _DEBUG
  #define SCOPED_GPU_MARKER(cmdList_, order_, name_) ScopedGpuMarker scopedGpuMarker(cmdList_, order_, name_);

  #define BEGIN_GPU_MARKER(cmdList_, order_, name_) \
    { \
      BeginGpuMarkerCmd cmd; \
      cmd.SetMarkerName(name_); \
      cmdList_->AddGpuCmd(cmd, order_); \
    }

  #define END_GPU_MARKER(cmdList_, order_) \
    { \
      EndGpuMarkerCmd cmd; \
      cmdList_->AddGpuCmd(cmd, order_); \
    }
#else
  #define SCOPED_GPU_MARKER(cmdList_, order_, name_)

  #define BEGIN_GPU_MARKER(cmdList_, order_, name_)

  #define END_GPU_MARKER(cmdList_, order_)
#endif

#endif 