#!/usr/bin/env python
# coding: utf-8

# # Install these Jupyter Lab Extensions
# 
# - @jupyter-widgets/jupyterlab-manager
# - vispy

# In[1]:


import ipywidgets as widgets
import numpy as np
import vispy
import vispy.gloo as gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate



n = 10000
a_position = np.random.uniform(-1, 1, (n, 3)).astype(np.float32)
a_color = np.random.uniform(0, 1, (n, 3)).astype(np.float32)
a_id = np.random.randint(0, 30, (n, 1))
a_id = np.sort(a_id, axis=0).astype(np.float32)

VERT_SHADER = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
attribute vec3 a_position;
attribute vec3 a_color;
attribute float a_id;
varying vec4 v_color;
varying float v_id;
void main (void) {
    v_id = a_id;
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    v_color = vec4(a_color, 1.0);
}
"""

FRAG_SHADER = """
varying float v_id;
varying vec4 v_color;
void main()
{
    float f = fract(v_id);
    // The second useless test is needed on OSX 10.8 (fuck)
    if( (f > 0.0001) && (f < .9999) )
        discard;
    else
        gl_FragColor = v_color;
}
"""

class Canvas(app.Canvas):

    # ---------------------------------
    def __init__(self, size=None, show=True):
        app.Canvas.__init__(self, keys='interactive', size=size)

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)

        # Set uniform and attribute
        self.program['a_id'] = gloo.VertexBuffer(a_id)
        self.program['a_position'] = gloo.VertexBuffer(a_position)
        self.program['a_color'] = gloo.VertexBuffer(a_color)
        #glsl_version=gloo.gl.glGetString(gloo.gl.GL_SHADING_LANGUAGE_VERSION)
        #print("glsl_version", glsl_version)

        self.translate = 5
        self.view = translate((0, 0, -self.translate), dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)

        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 1.0, 1000.0)
        self.program['u_projection'] = self.projection

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view

        self.theta = 0
        self.phi = 0

        self.context.set_clear_color('white')
        self.context.set_state('translucent')

        self.timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.slider1 = widgets.FloatSlider(value=0.1, min=0, max=1)
        self.slider2 = widgets.FloatSlider(value=0.1, min=0, max=1)

        if show:
            self.show()
            display(self.slider1)
            display(self.slider2)

    # ---------------------------------
    def on_key_press(self, event):
        
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()

    # ---------------------------------
    def on_timer(self, event):
        self.theta += self.slider1.value
        self.phi += self.slider2.value
        if True:
            global a_position, a_color, n
            a_position += (0.01*np.random.normal(0,1,a_position.shape))
            a_position = np.clip(a_position, (-1,-1,-1), (1,1,1))
            idcs=np.arange(n)
            w0,w1,w2 = 0.2,0.6,0.2
            a_color = (a_position+1)*0.5
            #a_color = w0*a_color[np.roll(idcs,-1)] + w1*a_color + w2*a_color[np.roll(idcs,+1)]
            a_color = np.clip(a_color, (0,0,0), (1,1,1))
            self.program['a_position'].set_data(a_position.astype(np.float32))
            self.program['a_color'].set_data(a_color.astype(np.float32))
        self.model = np.dot(rotate(self.theta, (0, 0, 1)),
                            rotate(self.phi, (0, 1, 0)))
        self.program['u_model'] = self.model
        self.update()

    # ---------------------------------
    def on_resize(self, event):
        gloo.set_viewport(0, 0, event.physical_size[0], event.physical_size[1])
        self.projection = perspective(45.0, event.size[0] /
                                      float(event.size[1]), 1.0, 1000.0)
        self.program['u_projection'] = self.projection

    # ---------------------------------
    def on_mouse_wheel(self, event):
        self.translate += event.delta[1]
        self.translate = max(2, self.translate)
        self.view = translate((0, 0, -self.translate))
        self.program['u_view'] = self.view
        self.update()

    # ---------------------------------
    def on_draw(self, event):
        self.context.clear()
        self.program.draw('line_strip')
        
c = Canvas(size=(640, 480))


# In[ ]:




