#ifndef BE_BIND_POINTS_H
#define BE_BIND_POINTS_H

/// Binds a shader resource to the given register.
#define bindpoint(reg) register(reg) : bindpoint_##reg

/// Binds a shader resource to the given register.
#define bindpoint_s(semantic, reg) register(reg) : semantic##__bindpoint_##reg

/// Excludes the given variable from resource binding.
#define prebound(name) unmanaged__##name

/// Excludes the given variable from resource binding.
#define prebound_s(semantic) semantic##__unmanaged

#endif