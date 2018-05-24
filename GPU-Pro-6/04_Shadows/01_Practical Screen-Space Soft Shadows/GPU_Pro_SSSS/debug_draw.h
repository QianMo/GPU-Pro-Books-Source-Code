#ifndef DEBUG_DRAW_H_
#define DEBUG_DRAW_H_

#include <GL/glew.h>
#include <mymath/mymath.h>
#include <vector>

using namespace std;
using namespace mymath;

class DebugDrawManager
{
  class dd_object
  {
  public:
    float lifetime_left;
    //0 means for one frame, any negative number means forever, any other number means lifetime in seconds
    dd_object(float lifetime):lifetime_left(lifetime){}

    virtual void Draw() const = 0;
  };

  class dd_sphere : public dd_object
  {
    vector< vec3 > vertices;
    vector< int > indices;
    
  public:
    dd_sphere(const mymath::vec3& POS, float r, float lifetime):dd_object(lifetime)
    {
      //GL_LINES
      const int resolution_u = 10;
      const int resolution_v = 10 *2;

      const float u_step = 1.f / resolution_u;
      const float v_step = 1.f / resolution_v;

      float u = 0;
      float v = 0;
      for(int u_index = 0; u_index < resolution_u; u_index += 1, u += u_step)
      {
        //0 <= alpha <= pi
        const float alpha = u*mymath::pi;

        for(int v_index = 0; v_index < resolution_v; v_index += 1, v += v_step)
        {
          //0 <= theta <= 2*pi
          const float theta = v*2*mymath::pi;
          vertices.push_back(POS + vec3(r*cos(alpha)*cos(theta), r*sin(alpha)*cos(theta), r*sin(theta)));
        }
      }

      int curr_index = 0;
      for(int u_index = 0; u_index < resolution_u; u_index += 1, u += u_step)
      {
        for(int v_index = 0; v_index < resolution_v; v_index += 1, v += v_step)
        {
          //if there is a preceding vertex
          if(v_index - 1 >= 0)
          {
            indices.push_back(curr_index);
          }

          indices.push_back(curr_index);

          curr_index += 1;
        }
        indices.push_back(curr_index - resolution_v);
      }
    }

    void Draw() const
    {
      glBegin( GL_LINES );
      for( auto& c : indices )
      {
        glVertex3f( vertices[c].x, vertices[c].y, vertices[c].z );
      }
      glEnd();
    }
  };

  class dd_cross : public dd_object
  {
    vector< vec3 > vertices;
    vector< int > indices;
    
  public:
    dd_cross(const mymath::vec3& POS, float size, float lifetime):dd_object(lifetime)
    {
      mymath::vec3 curr_start, curr_end;
      GLuint curr_index = 0;
      for(int i = 0; i < 3; i+=1)
      {
        curr_start = curr_end = POS;
        curr_start[i] -= size;
        vertices.push_back(curr_start);
        indices.push_back(curr_index++);

        curr_end[i] += size;
        vertices.push_back(curr_end);
        indices.push_back(curr_index++);
      }
    }

    void Draw() const
    {
      glBegin( GL_LINES );
      for( auto& c : indices )
      {
        glVertex3f( vertices[c].x, vertices[c].y, vertices[c].z );
      }
      glEnd();
    }
  };

  class dd_line_segment : public dd_object
  {
    vector< vec3 > vertices;
    vector< int > indices;
    
  public:
    dd_line_segment(const mymath::vec3& START, const mymath::vec3& END, float lifetime):dd_object(lifetime)
    {
      vertices.push_back(START);
      vertices.push_back(END);
      indices.push_back(0);
      indices.push_back(1);
    }

    void Draw() const
    {
      glBegin( GL_LINES );
      for( auto& c : indices )
      {
        glVertex3f( vertices[c].x, vertices[c].y, vertices[c].z );
      }
      glEnd();
    }
  };

  class dd_box : public dd_object
  {
    vector< vec3 > vertices;
    vector< int > indices;
  
  public:
    dd_box(const mymath::vec3& min, const mymath::vec3& max, float lifetime):dd_object(lifetime)
    {
      std::vector<mymath::vec3> min_max;
      min_max.push_back(min);
      min_max.push_back(max);

      for(int x_i = 0; x_i < 2; x_i += 1)
      {
        for(int y_i = 0; y_i < 2; y_i += 1)
        {
          for(int z_i = 0; z_i < 2; z_i += 1)
          {
            vertices.push_back(vec3(min_max[x_i].x, min_max[y_i].y, min_max[z_i].z));
          }
        }
      }

      //GL_LINES
      indices.push_back(0); indices.push_back(1);
      indices.push_back(0); indices.push_back(2);
      indices.push_back(0); indices.push_back(4);
      indices.push_back(1); indices.push_back(3);
      indices.push_back(1); indices.push_back(5);
      indices.push_back(2); indices.push_back(3);
      indices.push_back(2); indices.push_back(6);
      indices.push_back(3); indices.push_back(7);
      indices.push_back(4); indices.push_back(5);
      indices.push_back(4); indices.push_back(6);
      indices.push_back(5); indices.push_back(7);
      indices.push_back(6); indices.push_back(7);
    }

    void Draw() const
    {
      glBegin( GL_LINES );
      for( auto& c : indices )
      {
        glVertex3f( vertices[c].x, vertices[c].y, vertices[c].z );
      }
      glEnd();
    }
  };

  class dd_frustum : public dd_object
  {
    vector< vec3 > vertices;
    vector< int > indices;

  public:
    template< typename t >
    dd_frustum(const mymath::frame<t>& frame_to_draw, const mymath::vec3& pos, float scale, float lifetime):
    dd_object(lifetime)
    {
      vertices.push_back(frame_to_draw.near_ll.xyz*scale + pos);
      vertices.push_back(frame_to_draw.near_lr.xyz*scale + pos);
      vertices.push_back(frame_to_draw.near_ur.xyz*scale + pos);
      vertices.push_back(frame_to_draw.near_ul.xyz*scale + pos);

      vertices.push_back(frame_to_draw.far_ll.xyz*scale + pos);
      vertices.push_back(frame_to_draw.far_lr.xyz*scale + pos);
      vertices.push_back(frame_to_draw.far_ur.xyz*scale + pos);
      vertices.push_back(frame_to_draw.far_ul.xyz*scale + pos);

      //GL_LINES
      GLuint starting_index = 0;
      GLuint ending_index = 3;
      GLuint curr_index = starting_index;
      for(int i = 0; i < 2; i += 1)
      {
        for(int j = 0; j < 4; j += 1)
        {
          indicies.push_back(curr_index);
          indicies.push_back((curr_index == ending_index) ? starting_index : ++curr_index );
        }
        curr_index = starting_index = 4;
        ending_index = 7;
      }

      curr_index = 0;
      for(int i = 0; i < 4; i += 1)
      {
        indicies.push_back(curr_index);
        indicies.push_back(curr_index+4);
        curr_index += 1;
      }
    }

    void Draw() const
    {
      glBegin( GL_LINES );
      for( auto& c : indices )
      {
        glVertex3f( vertices[c].x, vertices[c].y, vertices[c].z );
      }
      glEnd();
    }
  };

  std::list<dd_object*> objects_to_draw;

public:
  void CreateLineSegment(const mymath::vec3& START, const mymath::vec3& END, float lifetime)
  {
    objects_to_draw.push_back(new dd_line_segment(START, END, lifetime));
  }

  void CreateCross(const mymath::vec3& POS, float size, float lifetime)
  {
    objects_to_draw.push_back(new dd_cross(POS, size, lifetime));
  }

  void CreateSphere(const mymath::vec3& POS, float radius, float lifetime)
  {
    objects_to_draw.push_back(new dd_sphere(POS, radius, lifetime));
  }

  void CreateAABoxMinMax(const mymath::vec3& min, const mymath::vec3& max, float lifetime)
  {
    objects_to_draw.push_back(new dd_box(min, max, lifetime));
  }

  void CreateAABoxPosEdges(const mymath::vec3& pos, const mymath::vec3& edge_halves, float lifetime)
  {
    const mymath::vec3 min = pos - edge_halves;
    const mymath::vec3 max = pos + edge_halves;
    objects_to_draw.push_back(new dd_box(min, max, lifetime));
  }

  template<typename t>
  void CreateFrustum(const mymath::frame<t>& frame_to_draw, const mymath::vec3& pos, float scale, float lifetime)
  {
    objects_to_draw.push_back(new dd_frustum(frame_to_draw, pos, scale, lifetime));
  }

  void DrawAndUpdate(float delta_time_sec)
  {
    //assumes that a proper shader is bound
    for(auto curr_obj_iter = objects_to_draw.begin(); curr_obj_iter != objects_to_draw.end(); ++curr_obj_iter)
    {
      dd_object* curr_obj = *curr_obj_iter;
      curr_obj->Draw();
    }

    for(auto curr_obj_iter = objects_to_draw.begin(); curr_obj_iter != objects_to_draw.end(); ++curr_obj_iter)
    {
      dd_object* curr_obj = *curr_obj_iter;
      if(0 <= curr_obj->lifetime_left)
      {
        curr_obj->lifetime_left -= delta_time_sec;
        if(0 >= curr_obj->lifetime_left)
        {
          delete curr_obj;
          curr_obj_iter = objects_to_draw.erase(curr_obj_iter);
          curr_obj_iter = objects_to_draw.begin();

          if(!objects_to_draw.size())
            return;
        }
      }
    }
  }

};

#endif /* DEBUG_DRAW_H_ */
