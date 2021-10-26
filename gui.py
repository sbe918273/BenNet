from loss import LOSS_FUNCS
from activation import ACTIVATION_FUNCS
from optimiser import gradient, OPTIMISER_REQUIREMENTS, OPTIMISER_FUNCS
from main import Network
from time import sleep
from itertools import chain

import tkinter as tk
import numpy as np

network = Network(input_dim=3, loss='sum_of_squares')
network.add_layer(4, 'sgd', 'relu')
network.add_layer(1, 'adam', 'sigmoid')

X = np.array([
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1],
])

y = np.array([
    [0, 0, 1, 0, 0, 1],
])

X_test = np.array([
    [0, 1, 0],
    [0, 1, 1],
    [0, 0, 1]
])

units = network.units
max_units = max(units)
get_height_buffer = lambda num_nodes: 0.5 * np.exp(-1 * (((1.5 * num_nodes) / max_units) ** 2))

def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)

def normalize(x, max_x=None, min_x=None):

    max_x = np.amax(x) if (max_x is None) else max_x
    min_x = np.amin(x) if (min_x is None) else min_x

    if min_x == max_x:
        return 0

    return ((x - min_x) / (max_x - min_x))

def arb_abs_normalize(x, arb_x=None, max_x=None):
    
    if max_x is None:

        flat_abs_x = list(chain(*arb_x))
        flat_abs_x = np.array([abs(i[0]) for i in flat_abs_x])

        max_x = np.amax(flat_abs_x) if (max_x) is None else max_x

    pos_norm = x.clip(min=0)
    pos_norm = normalize(pos_norm, max_x=max_x, min_x=0)

    neg_norm = -1 * x.clip(max=0)
    neg_norm = normalize(neg_norm, max_x=max_x, min_x=0)

    x_norm = np.clip(pos_norm - neg_norm, -1, 1)

    return x_norm, max_x

def get_colour(w):

    if w >= 0:
        colour_hex = hex(int(w * 4095))[2:].zfill(3)
        colour = "#000{}000".format(colour_hex)

    else:
        colour_hex = hex(int(w * -4095))[2:].zfill(3)
        colour = "#{}000000".format(colour_hex)

    if colour == "#0001007000":
        print(w)

    return colour

get_colour_vect = np.vectorize(get_colour)

tk.Canvas.create_circle = _create_circle

HEIGHT = 800
WIDTH = 1000
WIDTH_BUFFER = 50

root = tk.Tk()

canvas = tk.Canvas(root, bg="white", height=HEIGHT, width=WIDTH)

def draw_nodes(canvas, layers):

    def get_node_coords(units, i):

        node_coords = []

        height_buffer = (get_height_buffer(units - 1) * HEIGHT)
        window_height = HEIGHT - (height_buffer * 2)

        if units == 1:
            height_space = 0
            
        else:
            height_space = window_height / (units - 1)

        x = WIDTH_BUFFER + width_space * i

        for j in range(units):

            y = height_buffer + height_space * j

            node_coords.append((x, y))

        return node_coords

    num_layers = len(layers) + 1
    node_coords = []

    window_width = WIDTH - (WIDTH_BUFFER * 2)
    width_space = window_width / (num_layers - 1)

    arb_a = [layers[i].a for i in range(0, num_layers - 1)]

    max_a = None

    for i, layer in enumerate(layers):


        if max_a is None:
            a_norm, max_a = arb_abs_normalize(layer.a, arb_x=arb_a)

        else:
            a_norm, _ = arb_abs_normalize(layer.a, max_x=max_a)

        units = layer.units

        if type(a_norm) == int:
            a_norm = np.array([[a_norm]])

        colours = get_colour_vect(a_norm)

        node_coords.append(get_node_coords(units, i+1))

        for j, node_coord in enumerate(node_coords[i]):
            canvas.create_circle(*node_coord, 10, fill=colours[j][0])

    node_coords.insert(0, get_node_coords(3, 0))
    for j, node_coord in enumerate(node_coords[0]):
        canvas.create_circle(*node_coord, 10)
    
    return node_coords

def draw_lines(canvas, node_coords, layers):

    num_layers = len(node_coords)

    arb_w = [layers[i].w.reshape(-1, 1) for i in range(0, num_layers - 1)]

    max_w = None

    for i in range(0, num_layers - 1):

        if max_w is None:
            w_norm, max_w = arb_abs_normalize(layers[i].w, arb_x=arb_w)

        else:
            w_norm, _ = arb_abs_normalize(layers[i].w, max_x=max_w)

        colours = get_colour_vect(w_norm)
        widths = np.abs(w_norm) * 5

        for j, node in enumerate(node_coords[i]):

            for k, next_node in enumerate(node_coords[i + 1]):

                canvas.create_line(*node, *next_node, fill=colours[k, j], width=widths[k, j])

def update_canvas(canvas, network):

    canvas.delete("all")

    node_coords = draw_nodes(canvas, network.layers)
    draw_lines(canvas, node_coords, network.layers)
    canvas.pack()
    root.update()


update_canvas_ = lambda: update_canvas(canvas, network)
network.train(X, y, epochs=1000, batch_size=1, update=update_canvas_)

print(network.predict(X_test))
#update_canvas(canvas, network.layers, lines)



root.mainloop()



