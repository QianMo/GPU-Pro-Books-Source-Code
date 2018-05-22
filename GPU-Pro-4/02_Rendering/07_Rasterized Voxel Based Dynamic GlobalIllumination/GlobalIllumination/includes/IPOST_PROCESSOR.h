#ifndef IPOST_PROCESSOR_H
#define IPOST_PROCESSOR_H

class DX11_RENDER_TARGET;

// IPOST_PROCESSOR
//   Interface for post-processors.
class IPOST_PROCESSOR
{
public:
  IPOST_PROCESSOR()
	{
		active = true;
	}

  virtual ~IPOST_PROCESSOR()
	{
	}

	virtual bool Create()=0;

	virtual DX11_RENDER_TARGET* GetOutputRT() const=0;

	virtual void AddSurfaces()=0;

	const char* GetName() const
	{
		return name;
	}

	void SetActive(bool active)
	{
		this->active = active;
	}

	bool IsActive() const
	{
		return active;
	}

protected:
	char name[DEMO_MAX_STRING];
	bool active; 

};

#endif
