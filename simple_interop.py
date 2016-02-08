import pyopencl as cl
from pyopencl.tools import get_gl_sharing_context_properties
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy
import sys
import re

mf = cl.mem_flags

image_width = 0
image_height = 0
image_header = ""
image_maxval = 0

tex = 0
buf = 0
tex1 = 0

queue = 0
prog = 0
ctx = 0

image = 0


src = """

__kernel void add(__global char* a, __global char* b)
{
    unsigned int i = get_global_id(0);

    b[i] = a[i] + 100;
}

"""


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)

    global image_width
    image_width = int(width)
    global image_height
    image_height = int(height)


    return numpy.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


def init():
    platform = cl.get_platforms()[0]
    global ctx
    from pyopencl.tools import get_gl_sharing_context_properties
    import sys
    if sys.platform == "darwin":
        ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                devices=[])
    else:
        # Some OSs prefer clCreateContextFromType, some prefer
        # clCreateContext. Try both.
        try:
            ctx = cl.Context(properties=[
                (cl.context_properties.PLATFORM, platform)]
                + get_gl_sharing_context_properties())
        except:
            ctx = cl.Context(properties=[
                (cl.context_properties.PLATFORM, platform)]
                + get_gl_sharing_context_properties(),
                devices = [platform.get_devices()[0]])

    global buf
    global tex
    global tex1
    global image

    image = read_pgm("lena512.pgm", byteorder='<')
    image = image.ravel() # make 1D
    size = image_height * image_width


    #Texture setup base
    tex1 = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex1)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, image_width, image_height, 0, GL_RED, GL_UNSIGNED_BYTE, image);



    # Texture setup CL
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, image_width, image_height, 0, GL_RED, GL_UNSIGNED_BYTE, None);


    # Buffer setup
    buf = glGenBuffers(1)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf)
    glBufferData(GL_PIXEL_UNPACK_BUFFER, image_height * image_width, None, GL_STREAM_DRAW)


    # CL setup
    global queue
    global prog

    #image_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
    #mem = cl.GLBuffer(ctx, mf.WRITE_ONLY, numpy.float32(buf))

    queue = cl.CommandQueue(ctx)
    prog = cl.Program(ctx, src).build()

    #cl.enqueue_acquire_gl_objects(queue, [mem])
    #add_knl = prog.add
    #add_knl.set_args(image_buf, mem)
    #cl.enqueue_nd_range_kernel(queue, add_knl, image.shape, None)
    #cl.enqueue_release_gl_objects(queue, [mem])

    #queue.finish()
    #glFlush()


    # Unbind
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);


def change_display(image) :

    image_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
    mem = cl.GLBuffer(ctx, mf.WRITE_ONLY, numpy.float32(buf))

    cl.enqueue_acquire_gl_objects(queue, [mem])
    add_knl = prog.add
    add_knl.set_args(image_buf, mem)
    cl.enqueue_nd_range_kernel(queue, add_knl, image.shape, None)
    cl.enqueue_release_gl_objects(queue, [mem])

    queue.finish()
    glFlush()


def glut_window():

    width = 1024 #512
    height = 512
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow("Display Example")

    glutDisplayFunc(on_display)

    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

# add CL stuff
def on_display():

    change_display(image)

    glClear(GL_COLOR_BUFFER_BIT)

    glBindTexture(GL_TEXTURE_2D, tex)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RED, GL_UNSIGNED_BYTE, None)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    glEnable(GL_TEXTURE_2D)

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, -1.0, 0.0)
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, 1.0, 0.0)
    glTexCoord2f(1.0, .0); glVertex3f(0.0, 1.0, 0.0)
    glTexCoord2f(1.0, 1.0); glVertex3f(0.0, -1.0, 0.0)
    glEnd()

    glBindTexture(GL_TEXTURE_2D, tex1)

    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 1.0); glVertex3f(0.0, -1.0, 0.0)
    glTexCoord2f(0.0, 0.0); glVertex3f(0.0, 1.0, 0.0)
    glTexCoord2f(1.0, .0); glVertex3f(1.0, 1.0, 0.0)
    glTexCoord2f(1.0, 1.0); glVertex3f(1.0, -1.0, 0.0)
    glEnd()

    glBindTexture(GL_TEXTURE_2D, 0)

    glDisable(GL_TEXTURE_2D)

    glutSwapBuffers()
    glutPostRedisplay()




glut_window()
init()
glutMainLoop()
