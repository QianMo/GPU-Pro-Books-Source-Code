/*
    Copyright 2005-2008 Intel Corporation.  All Rights Reserved.

    This file is part of Threading Building Blocks.

    Threading Building Blocks is free software; you can redistribute it
    and/or modify it under the terms of the GNU General Public License
    version 2 as published by the Free Software Foundation.

    Threading Building Blocks is distributed in the hope that it will be
    useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Threading Building Blocks; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

    As a special exception, you may use this file as part of a free software
    library without restriction.  Specifically, if other files instantiate
    templates or use macros or inline functions from this file, or you compile
    this file and link it with other files to produce an executable, this
    file does not by itself cause the resulting executable to be covered by
    the GNU General Public License.  This exception does not however
    invalidate any other reasons why the executable file might be covered by
    the GNU General Public License.
*/

#ifndef __TBB_pipeline_H 
#define __TBB_pipeline_H 

#include "atomic.h"
#include "task.h"
#include <cstddef>

namespace tbb {

class pipeline;
class filter;

//! @cond INTERNAL
namespace internal {
const unsigned char IS_SERIAL = 0x1;
const unsigned char SERIAL_MODE_MASK = 0x1; // the lowest bit 0 is for parallel vs. serial 

// The argument for PIPELINE_VERSION should be an integer between 2 and 9
#define __TBB_PIPELINE_VERSION(x) (unsigned char)(x-2)<<1
const unsigned char VERSION_MASK = 0x7<<1; // bits 1-3 are for version
const unsigned char CURRENT_VERSION = __TBB_PIPELINE_VERSION(3);

typedef unsigned long Token;
typedef long tokendiff_t;
class stage_task;
class ordered_buffer;

} // namespace internal
//! @endcond

//! A stage in a pipeline.
/** @ingroup algorithms */
class filter {
private:
    //! Value used to mark "not in pipeline"
    static filter* not_in_pipeline() {return reinterpret_cast<filter*>(internal::intptr(-1));}
protected:
    //! For pipeline version 2 and earlier 0 is parallel and 1 is serial mode
    enum mode {
        parallel = internal::CURRENT_VERSION,
        serial = internal::CURRENT_VERSION | internal::IS_SERIAL
    };

    filter( bool is_serial_ ) : 
        next_filter_in_pipeline(not_in_pipeline()),
        input_buffer(NULL),
        my_filter_mode(static_cast<unsigned char>(is_serial_ ? serial : parallel)),
        prev_filter_in_pipeline(not_in_pipeline()),
        my_pipeline(NULL)
    {}
    
    filter( mode filter_mode ) :
        next_filter_in_pipeline(not_in_pipeline()),
        input_buffer(NULL),
        my_filter_mode(static_cast<unsigned char>(filter_mode)),
        prev_filter_in_pipeline(not_in_pipeline()),
        my_pipeline(NULL)
    {}


public:
    //! True if filter must receive stream in order.
    bool is_serial() const {
        return (my_filter_mode & internal::SERIAL_MODE_MASK) == internal::IS_SERIAL;
    }  

    //! Operate on an item from the input stream, and return item for output stream.
    /** Returns NULL if filter is a sink. */
    virtual void* operator()( void* item ) = 0;

    //! Destroy filter.  
    /** If the filter was added to a pipeline, the pipeline must be destroyed first. */
    virtual ~filter();

private:
    //! Pointer to next filter in the pipeline.
    filter* next_filter_in_pipeline;

    //! Input buffer for filter that requires serial input; NULL otherwise. */
    internal::ordered_buffer* input_buffer;

    friend class internal::stage_task;
    friend class pipeline;

    //! Internal storage for is_serial()
    const unsigned char my_filter_mode;

    //! Pointer to previous filter in the pipeline.
    filter* prev_filter_in_pipeline;

    //! Pointer to the pipeline
    pipeline* my_pipeline;
};

//! A processing pipeling that applies filters to items.
/** @ingroup algorithms */
class pipeline {
public:
    //! Construct empty pipeline.
    pipeline();

    //! Destroy pipeline.
    virtual ~pipeline();

    //! Add filter to end of pipeline.
    void add_filter( filter& filter_ );

    //! Run the pipeline to completion.
    void run( size_t max_number_of_live_tokens );

    //! Remove all filters from the pipeline
    void clear();

private:
    friend class internal::stage_task;
    friend class filter;

    //! Pointer to first filter in the pipeline.
    filter* filter_list;

    //! Pointer to location where address of next filter to be added should be stored.
    filter* filter_end;

    //! task who's reference count is used to determine when all stages are done.
    empty_task* end_counter;

    //! Number of idle tokens waiting for input stage.
    atomic<internal::Token> input_tokens;

    //! Number of tokens created so far.
    internal::Token token_counter;

    //! False until fetch_input returns NULL.
    bool end_of_input;

    //! Remove filter from pipeline.
    void remove_filter( filter& filter_ );

    //! Not used, but retained to satisfy old export files.
    void inject_token( task& self );
};

} // tbb

#endif /* __TBB_pipeline_H */
