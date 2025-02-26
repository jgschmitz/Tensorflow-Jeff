#nvidia

#watch -n 1 nvidia-smi to monitor memory usage every second.

#Often, extra Python processes can stay running in the background, maintaining a hold on the GPU memory, even if nvidia-smi doesn't show it.
#Probably due to running Keras in a notebook, and then running the cell that starts the processes again, since this will fork the current process, which has a hold on GPU memory. In the future, restart the kernel first, and stop all process before exiting (even though they are daemons and should stop automatically when the parent process ends).
#Kill old Python process with ps -ef | grep `whoami` | grep "[p]ython" | awk '{print $2}' | xargs kill -9.
#Quick GPU test

python3

import tensorflow as tf
a = tf.constant(3)
b = tf.constant(4)
c = a * b
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
  sess.run(c)
Check for GPU device creation and placement of operations, i.e. (2 GPU devices here):

I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:83:00.0)
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:1) -> (device: 1, name: Tesla K80, pci bus id: 0000:84:00.0)
Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: Tesla K80, pci bus id: 0000:83:00.0
/job:localhost/replica:0/task:0/gpu:1 -> device: 1, name: Tesla K80, pci bus id: 0000:84:00.0
I tensorflow/core/common_runtime/direct_session.cc:255] Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: Tesla K80, pci bus id: 0000:83:00.0
/job:localhost/replica:0/task:0/gpu:1 -> device: 1, name: Tesla K80, pci bus id: 0000:84:00.0

mul: (Mul): /job:localhost/replica:0/task:0/gpu:0
I tensorflow/core/common_runtime/simple_placer.cc:827] mul: (Mul)/job:localhost/replica:0/task:0/gpu:0
Const_1: (Const): /job:localhost/replica:0/task:0/gpu:0
I tensorflow/core/common_runtime/simple_placer.cc:827] Const_1: (Const)/job:localhost/replica:0/task:0/gpu:0
Const: (Const): /job:localhost/replica:0/task:0/gpu:0
I tensorflow/core/common_runtime/simple_placer.cc:827] Const: (Const)/job:localhost/replica:0/task:0/gpu:0
Limit Available GPUs
Two options:

Environment variable CUDA_VISIBLE_DEVICES equal to numeric IDs of GPUs to be made available. Can either set this at the command line, or with Python using the following:
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
TensorFlow API
c = tf.ConfigProto()
c.gpu_options.visible_device_list="0"
sess = tf.Session(config=c)
TensorBoard
tensorboard --logdir /path/to/logs --host HOSTNAME_OR_IP --port PORT

Visualize TensorBoard graph within a Jupyter notebook:
(original source here):
# From: http://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
# Helper functions for TF Graph visualization
from IPython.display import clear_output, Image, display, HTML
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = bytes("<stripped %d bytes>"%size, 'utf-8')
    return strip_def
  
def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add() 
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def
  
def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))
  
    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

# Visualizing the network graph. Be sure expand the "mixed" nodes to see their 
# internal structure. We are going to visualize "Conv2D" nodes.
graph_def = tf.get_default_graph().as_graph_def()
tmp_def = rename_nodes(graph_def, lambda s:"/".join(s.split('_',1)))
show_graph(tmp_def)
