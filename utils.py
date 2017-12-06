from PIL import ImageDraw
import torch
import numpy as np

class VisdomLinePlotter:
    """Plots to Visdom"""
    def __init__(self, plot_name, y_axis='loss', env_name='main'):
        import visdom
        self.viz = visdom.Visdom()
        self.env = env_name
        self.plot_name = plot_name
        self.y_axis = y_axis
        self.window = None

    def plot(self, var_name, split_name, x, y):
        name = "{} {}".format(var_name, split_name)
        if self.window is None:
            self.window = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[name],
                title=self.plot_name,
                xlabel='Epochs',
                ylabel=self.y_axis
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.window, name=name)

def draw_matches(image, multiscan_matches):
    "draw scan_multiple_scales() results on the source image"
    im_copy = image.copy()
    draw = ImageDraw.Draw(im_copy)
    for matches, _, _ in multiscan_matches:
        for match in matches:
            draw.rectangle([int(i) for i in match[:4]], outline=(255,0,0,255))
    return im_copy

def to_fddb_ellipses(boxes):
    centers = (boxes[:,0:2] + boxes[:,2:4]) * 0.5
    dims = (boxes[:,2:4] - boxes[:,0:2]) * 0.5 # radii so halve
    axes = dims * torch.Tensor([1.13, 1.18])

    # major axis, minor axis, angle, center x, center y, score
    ellipses = torch.cat([axes, torch.zeros(boxes.size(0), 1), centers, boxes[:,4]], dim=1)
    # stringify
    return ["{} {} {} {} {} {}".format(*row) for row in ellipses]

# set of scales for scanning an image
scan_scales = [1.18**(-i) for i in range(3,20,2)]