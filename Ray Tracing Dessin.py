"""
Projet Ray-Tracing BA3
Script Dessin

De Saulnier kristopher et Latte Samuel
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import constants
import matplotlib as mpl


# Il faut s'assurer que ces paramètres soient communs entre le main et le dessin
Emetteur = [np.array([19,1]),np.array([41,7])]  # Position des émetteurs
nbr_rebond = 2
resol_x = 0.2 # résolution en x
resol_y = 0.2 # résolution en y


# On commence par définir les murs et les couleurs :
murs = {'mur1': [np.array([0+12,0+8]),np.array([31+12,0+8])],
        'mur2': [np.array([31+12,0+8]),np.array([32+12,-1+8])],
        'mur3': [np.array([32+12,-1+8]),np.array([32+12,-7+8])],
        'mur4': [np.array([32+12,-7+8]),np.array([31+12,-8+8])],
        'mur5': [np.array([31+12,-8+8]),np.array([0+12,-8+8])],

        # bas du nez
        'mur6': [np.array([0+12,-8+8]),np.array([-4+12,-8 + (4*np.tan(3.75*(constants.pi/180)))+8])],
        'mur7': [np.array([-4+12,-8 + (4*np.tan(3.75*(constants.pi/180)))+8]), np.array([-7+12,-8 + (4*np.tan(3.75*(constants.pi/180))) + (3*np.tan(7.5*(constants.pi/180)))+8])],
        'mur8': [np.array([-7+12,-8 + (4*np.tan(3.75*(constants.pi/180))) + (3*np.tan(7.5*(constants.pi/180)))+8]),np.array([-9+12,-8 + (4*np.tan(3.75*(constants.pi/180))) + (3*np.tan(7.5*(constants.pi/180))) + (2*np.tan(15*(constants.pi/180)))+8])],
        'mur9': [np.array([-9+12,-8 + (4*np.tan(3.75*(constants.pi/180))) + (3*np.tan(7.5*(constants.pi/180))) + (2*np.tan(15*(constants.pi/180)))+8]), np.array([-10.5+12, -8 + (4*np.tan(3.75*(constants.pi/180))) + (3*np.tan(7.5*(constants.pi/180))) + (2*np.tan(15*(constants.pi/180))) + (1.5*np.tan(30*(constants.pi/180)))+8])],
        'mur10': [np.array([-10.5+12, -8 + (4*np.tan(3.75*(constants.pi/180))) + (3*np.tan(7.5*(constants.pi/180))) + (2*np.tan(15*(constants.pi/180))) + (1.5*np.tan(30*(constants.pi/180)))+8]), np.array([-11.5+12, -8 + (4*np.tan(3.75*(constants.pi/180))) + (3*np.tan(7.5*(constants.pi/180))) + (2*np.tan(15*(constants.pi/180))) + (1.5*np.tan(30*(constants.pi/180))) + np.tan(45*(constants.pi/180))+8])],
        'mur11': [np.array([-11.5+12, -8 + (4*np.tan(3.75*(constants.pi/180))) + (3*np.tan(7.5*(constants.pi/180))) + (2*np.tan(15*(constants.pi/180))) + (1.5*np.tan(30*(constants.pi/180))) + np.tan(45*(constants.pi/180))+8]), np.array([-12+12, -8 + (4*np.tan(3.75*(constants.pi/180))) + (3*np.tan(7.5*(constants.pi/180))) + (2*np.tan(15*(constants.pi/180))) + (1.5*np.tan(30*(constants.pi/180))) + np.tan(45*(constants.pi/180)) + (0.5*np.tan(60*(constants.pi/180)))+8])],
        'mur12': [np.array([-12+12, -8 + (4*np.tan(3.75*(constants.pi/180))) + (3*np.tan(7.5*(constants.pi/180))) + (2*np.tan(15*(constants.pi/180))) + (1.5*np.tan(30*(constants.pi/180))) + np.tan(45*(constants.pi/180)) + (0.5*np.tan(60*(constants.pi/180)))+8]), np.array([-12+12, -8 + (4*np.tan(3.75*(constants.pi/180))) + (3*np.tan(7.5*(constants.pi/180))) + (2*np.tan(15*(constants.pi/180))) + (1.5*np.tan(30*(constants.pi/180))) + np.tan(45*(constants.pi/180)) + (0.5*np.tan(60*(constants.pi/180))) + 0.15+8])],

        #haut du nez
        'mur13': [np.array([0+12,0+8]), np.array([-4+12, - (4*np.tan(3.75*(constants.pi/180)))+8])],
        'mur14': [np.array([-4+12, - (4*np.tan(3.75*(constants.pi/180)))+8]), np.array([-7+12, - (4*np.tan(3.75*(constants.pi/180))) - (3*np.tan(7.5*(constants.pi/180)))+8])],
        'mur15': [np.array([-7+12, - (4*np.tan(3.75*(constants.pi/180))) - (3*np.tan(7.5*(constants.pi/180)))+8]), np.array([-9+12, - (4*np.tan(3.75*(constants.pi/180))) - (3*np.tan(7.5*(constants.pi/180))) - (2*np.tan(15*(constants.pi/180)))+8])],
        'mur16': [np.array([-9+12, - (4*np.tan(3.75*(constants.pi/180))) - (3*np.tan(7.5*(constants.pi/180))) - (2*np.tan(15*(constants.pi/180)))+8]), np.array([-10.5+12, - (4*np.tan(3.75*(constants.pi/180))) - (3*np.tan(7.5*(constants.pi/180))) - (2*np.tan(15*(constants.pi/180))) - (1.5*np.tan(30*(constants.pi/180)))+8])],
        'mur17': [np.array([-10.5+12, - (4*np.tan(3.75*(constants.pi/180))) - (3*np.tan(7.5*(constants.pi/180))) - (2*np.tan(15*(constants.pi/180))) - (1.5*np.tan(30*(constants.pi/180)))+8]), np.array([-11.5+12, - (4*np.tan(3.75*(constants.pi/180))) - (3*np.tan(7.5*(constants.pi/180))) - (2*np.tan(15*(constants.pi/180))) - (1.5*np.tan(30*(constants.pi/180))) - np.tan(45*(constants.pi/180))+8])],
        'mur18': [np.array([-11.5+12, - (4*np.tan(3.75*(constants.pi/180))) - (3*np.tan(7.5*(constants.pi/180))) - (2*np.tan(15*(constants.pi/180))) - (1.5*np.tan(30*(constants.pi/180))) - np.tan(45*(constants.pi/180))+8]), np.array([-12+12, - (4*np.tan(3.75*(constants.pi/180))) - (3*np.tan(7.5*(constants.pi/180))) - (2*np.tan(15*(constants.pi/180))) - (1.5*np.tan(30*(constants.pi/180))) - np.tan(45*(constants.pi/180)) - (0.5*np.tan(60*(constants.pi/180)))+8])],
        'mur19': [np.array([-12+12, - (4*np.tan(3.75*(constants.pi/180))) - (3*np.tan(7.5*(constants.pi/180))) - (2*np.tan(15*(constants.pi/180))) - (1.5*np.tan(30*(constants.pi/180))) - np.tan(45*(constants.pi/180)) - (0.5*np.tan(60*(constants.pi/180)))+8]), np.array([-12+12, - (4*np.tan(3.75*(constants.pi/180))) - (3*np.tan(7.5*(constants.pi/180))) - (2*np.tan(15*(constants.pi/180))) - (1.5*np.tan(30*(constants.pi/180))) - np.tan(45*(constants.pi/180)) - (0.5*np.tan(60*(constants.pi/180))) - 0.15+8])],

        #séparation entre cockpit-sièges, entre classes et parois du fond de l'avion
        'mur20': [np.array([-4+12, - (4*np.tan(3.75*(constants.pi/180)))+8]), np.array([-4+12,-8 + (4*np.tan(3.75*(constants.pi/180)))+8])],
        'mur21': [np.array([0+12, 0+8]), np.array([0+12, -3+8])],
        'mur22': [np.array([0+12, -8+8]), np.array([0+12, -5+8])],
        'mur23': [np.array([8+12, 0+8]), np.array([8+12, -3+8])],
        'mur24': [np.array([8+12, -8+8]), np.array([8+12, -5+8])],
        'mur25': [np.array([26+12, 0+8]), np.array([26+12, -3+8])],
        'mur26': [np.array([26+12, -8+8]), np.array([26+12, -5+8])],
        'mur27': [np.array([28+12, 0+8]), np.array([28+12, -3+8])],
        'mur28': [np.array([28+12, -8+8]), np.array([28+12, -5+8])],

        #sièges de droite
        'mur29': [np.array([2+12, 0+8]), np.array([2+12, -3+8])],
        'mur30': [np.array([4+12, 0+8]), np.array([4+12, -3+8])],
        'mur31': [np.array([6+12, 0+8]), np.array([6+12, -3+8])],
        'mur32': [np.array([10+12, 0+8]), np.array([10+12, -3+8])],
        'mur33': [np.array([11.5+12, 0+8]), np.array([11.5+12, -3+8])],
        'mur34': [np.array([13+12, 0+8]), np.array([13+12, -3+8])],
        'mur35': [np.array([14.5+12, 0+8]), np.array([14.5+12, -3+8])],
        'mur36': [np.array([16+12, 0+8]), np.array([16+12, -3+8])],
        'mur37': [np.array([17.5+12, 0+8]), np.array([17.5+12, -3+8])],
        'mur38': [np.array([19+12, 0+8]), np.array([19+12, -3+8])],
        'mur39': [np.array([20.5+12, 0+8]), np.array([20.5+12, -3+8])],
        'mur40': [np.array([22+12, 0+8]), np.array([22+12, -3+8])],
        'mur41': [np.array([23.5+12, 0+8]), np.array([23.5+12, -3+8])],
        'mur42': [np.array([25+12, 0+8]), np.array([25+12, -3+8])],

        #sièges de gauche
        'mur43': [np.array([2+12, -8+8]), np.array([2+12, -5+8])],
        'mur44': [np.array([4+12, -8+8]), np.array([4+12, -5+8])],
        'mur45': [np.array([6+12, -8+8]), np.array([6+12, -5+8])],
        'mur46': [np.array([10+12, -8+8]), np.array([10+12, -5+8])],
        'mur47': [np.array([11.5+12, -8+8]), np.array([11.5+12, -5+8])],
        'mur48': [np.array([13+12, -8+8]), np.array([13+12, -5+8])],
        'mur49': [np.array([14.5+12, -8+8]), np.array([14.5+12, -5+8])],
        'mur50': [np.array([16+12, -8+8]), np.array([16+12, -5+8])],
        'mur51': [np.array([17.5+12, -8+8]), np.array([17.5+12, -5+8])],
        'mur52': [np.array([19+12, -8+8]), np.array([19+12, -5+8])],
        'mur53': [np.array([20.5+12, -8+8]), np.array([20.5+12, -5+8])],
        'mur54': [np.array([22+12, -8+8]), np.array([22+12, -5+8])],
        'mur55': [np.array([23.5+12, -8+8]), np.array([23.5+12, -5+8])],
        'mur56': [np.array([25+12, -8+8]), np.array([25+12, -5+8])]}

colors_rgb = [
    [0,         0,    0.5156],
    [0,         0,    0.5312],
    [0,         0,    0.5469],
    [0,         0,    0.5625],
    [0,         0,    0.5781],
    [0,         0,    0.5938],
    [0,         0,    0.6094],
    [0,         0,    0.6250],
    [0,         0,    0.6406],
    [0,         0,    0.6562],
    [0,         0,    0.6719],
    [0,         0,    0.6875],
    [0,         0,    0.7031],
    [0,         0,    0.7188],
    [0,         0,    0.7344],
    [0,         0,    0.7500],
    [0,         0,    0.7656],
    [0,         0,    0.7812],
    [0,         0,    0.7969],
    [0,         0,    0.8125],
    [0,         0,    0.8281],
    [0,         0,    0.8438],
    [0,         0,    0.8594],
    [0,         0,    0.8750],
    [0,         0,    0.8906],
    [0,         0,    0.9062],
    [0,         0,    0.9219],
    [0,         0,    0.9375],
    [0,         0,    0.9531],
    [0,         0,    0.9688],
    [0,         0,    0.9844],
    [0,         0,    1.0000],
    [0,    0.0156,    1.0000],
    [0,    0.0312,    1.0000],
    [0,    0.0469,    1.0000],
    [0,    0.0625,    1.0000],
    [0,    0.0781,    1.0000],
    [0,    0.0938,    1.0000],
    [0,    0.1094,    1.0000],
    [0,    0.1250,    1.0000],
    [0,    0.1406,    1.0000],
    [0,    0.1562,    1.0000],
    [0,    0.1719,    1.0000],
    [0,    0.1875,    1.0000],
    [0,    0.2031,    1.0000],
    [0,    0.2188,    1.0000],
    [0,    0.2344,    1.0000],
    [0,    0.2500,    1.0000],
    [0,    0.2656,    1.0000],
    [0,    0.2812,    1.0000],
    [0,    0.2969,    1.0000],
    [0,    0.3125,    1.0000],
    [0,    0.3281,    1.0000],
    [0,    0.3438,    1.0000],
    [0,    0.3594,    1.0000],
    [0,    0.3750,    1.0000],
    [0,    0.3906,    1.0000],
    [0,    0.4062,    1.0000],
    [0,    0.4219,    1.0000],
    [0,    0.4375,    1.0000],
    [0,    0.4531,    1.0000],
    [0,    0.4688,    1.0000],
    [0,    0.4844,    1.0000],
    [0,    0.5000,    1.0000],
    [0,    0.5156,    1.0000],
    [0,    0.5312,    1.0000],
    [0,    0.5469,    1.0000],
    [0,    0.5625,    1.0000],
    [0,    0.5781,    1.0000],
    [0,    0.5938,    1.0000],
    [0,    0.6094,    1.0000],
    [0,    0.6250,    1.0000],
    [0,    0.6406,    1.0000],
    [0,    0.6562,    1.0000],
    [0,    0.6719,    1.0000],
    [0,    0.6875,    1.0000],
    [0,    0.7031,    1.0000],
    [0,    0.7188,    1.0000],
    [0,    0.7344,    1.0000],
    [0,    0.7500,    1.0000],
    [0,    0.7656,    1.0000],
    [0,    0.7812,    1.0000],
    [0,    0.7969,    1.0000],
    [0,    0.8125,    1.0000],
    [0,    0.8281,    1.0000],
    [0,    0.8438,    1.0000],
    [0,    0.8594,    1.0000],
    [0,    0.8750,    1.0000],
    [0,    0.8906,    1.0000],
    [0,    0.9062,    1.0000],
    [0,    0.9219,    1.0000],
    [0,    0.9375,    1.0000],
    [0,    0.9531,    1.0000],
    [0,    0.9688,    1.0000],
    [0,    0.9844,    1.0000],
    [0,    1.0000,    1.0000],
    [0.0156,    1.0000,    0.9844],
    [0.0312,    1.0000,    0.9688],
    [0.0469,    1.0000,    0.9531],
    [0.0625,    1.0000,    0.9375],
    [0.0781,    1.0000,    0.9219],
    [0.0938,    1.0000,    0.9062],
    [0.1094,    1.0000,    0.8906],
    [0.1250,    1.0000,    0.8750],
    [0.1406,    1.0000,    0.8594],
    [0.1562,    1.0000,    0.8438],
    [0.1719,    1.0000,    0.8281],
    [0.1875,    1.0000,    0.8125],
    [0.2031,    1.0000,    0.7969],
    [0.2188,    1.0000,    0.7812],
    [0.2344,    1.0000,    0.7656],
    [0.2500,    1.0000,    0.7500],
    [0.2656,    1.0000,    0.7344],
    [0.2812,    1.0000,    0.7188],
    [0.2969,    1.0000,    0.7031],
    [0.3125,    1.0000,    0.6875],
    [0.3281,    1.0000,    0.6719],
    [0.3438,    1.0000,    0.6562],
    [0.3594,    1.0000,    0.6406],
    [0.3750,    1.0000,    0.6250],
    [0.3906,    1.0000,    0.6094],
    [0.4062,    1.0000,    0.5938],
    [0.4219,    1.0000,    0.5781],
    [0.4375,    1.0000,    0.5625],
    [0.4531,    1.0000,    0.5469],
    [0.4688,    1.0000,    0.5312],
    [0.4844,    1.0000,    0.5156],
    [0.5000,    1.0000,    0.5000],
    [0.5156,    1.0000,    0.4844],
    [0.5312,    1.0000,    0.4688],
    [0.5469,    1.0000,    0.4531],
    [0.5625,    1.0000,    0.4375],
    [0.5781,    1.0000,    0.4219],
    [0.5938,    1.0000,    0.4062],
    [0.6094,    1.0000,    0.3906],
    [0.6250,    1.0000,    0.3750],
    [0.6406,    1.0000,    0.3594],
    [0.6562,    1.0000,    0.3438],
    [0.6719,    1.0000,    0.3281],
    [0.6875,    1.0000,    0.3125],
    [0.7031,    1.0000,    0.2969],
    [0.7188,    1.0000,    0.2812],
    [0.7344,    1.0000,    0.2656],
    [0.7500,    1.0000,    0.2500],
    [0.7656,    1.0000,    0.2344],
    [0.7812,    1.0000,    0.2188],
    [0.7969,    1.0000,    0.2031],
    [0.8125,    1.0000,    0.1875],
    [0.8281,    1.0000,    0.1719],
    [0.8438,    1.0000,    0.1562],
    [0.8594,    1.0000,    0.1406],
    [0.8750,    1.0000,    0.1250],
    [0.8906,    1.0000,    0.1094],
    [0.9062,    1.0000,    0.0938],
    [0.9219,    1.0000,    0.0781],
    [0.9375,    1.0000,    0.0625],
    [0.9531,    1.0000,    0.0469],
    [0.9688,    1.0000,    0.0312],
    [0.9844,    1.0000,    0.0156],
    [1.0000,    1.0000,         0],
    [1.0000,    0.9844,         0],
    [1.0000,    0.9688,         0],
    [1.0000,    0.9531,         0],
    [1.0000,    0.9375,         0],
    [1.0000,    0.9219,         0],
    [1.0000,    0.9062,         0],
    [1.0000,    0.8906,         0],
    [1.0000,    0.8750,         0],
    [1.0000,    0.8594,         0],
    [1.0000,    0.8438,         0],
    [1.0000,    0.8281,         0],
    [1.0000,    0.8125,         0],
    [1.0000,    0.7969,         0],
    [1.0000,    0.7812,         0],
    [1.0000,    0.7656,         0],
    [1.0000,    0.7500,         0],
    [1.0000,    0.7344,         0],
    [1.0000,    0.7188,         0],
    [1.0000,    0.7031,         0],
    [1.0000,    0.6875,         0],
    [1.0000,    0.6719,         0],
    [1.0000,    0.6562,         0],
    [1.0000,    0.6406,         0],
    [1.0000,    0.6250,         0],
    [1.0000,    0.6094,         0],
    [1.0000,    0.5938,         0],
    [1.0000,    0.5781,         0],
    [1.0000,    0.5625,         0],
    [1.0000,    0.5469,         0],
    [1.0000,    0.5312,         0],
    [1.0000,    0.5156,         0],
    [1.0000,    0.5000,         0],
    [1.0000,    0.4844,         0],
    [1.0000,    0.4688,         0],
    [1.0000,    0.4531,         0],
    [1.0000,    0.4375,         0],
    [1.0000,    0.4219,         0],
    [1.0000,    0.4062,         0],
    [1.0000,    0.3906,         0],
    [1.0000,    0.3750,         0],
    [1.0000,    0.3594,         0],
    [1.0000,    0.3438,         0],
    [1.0000,    0.3281,         0],
    [1.0000,    0.3125,         0],
    [1.0000,    0.2969,         0],
    [1.0000,    0.2812,         0],
    [1.0000,    0.2656,         0],
    [1.0000,    0.2500,         0],
    [1.0000,    0.2344,         0],
    [1.0000,    0.2188,         0],
    [1.0000,    0.2031,         0],
    [1.0000,    0.1875,         0],
    [1.0000,    0.1719,         0],
    [1.0000,    0.1562,         0],
    [1.0000,    0.1406,         0],
    [1.0000,    0.1250,         0],
    [1.0000,    0.1094,         0],
    [1.0000,    0.0938,         0],
    [1.0000,    0.0781,         0],
    [1.0000,    0.0625,         0],
    [1.0000,    0.0469,         0],
    [1.0000,    0.0312,         0],
    [1.0000,    0.0156,         0],
    [1.0000,         0,         0],
    [0.9844,         0,         0],
    [0.9688,         0,         0],
    [0.9531,         0,         0],
    [0.9375,         0,         0],
    [0.9219,         0,         0],
    [0.9062,         0,         0],
    [0.8906,         0,         0],
    [0.8750,         0,         0],
    [0.8594,         0,         0],
    [0.8438,         0,         0],
    [0.8281,         0,         0],
    [0.8125,         0,         0],
    [0.7969,         0,         0],
    [0.7812,         0,         0],
    [0.7656,         0,         0],
    [0.7500,         0,         0],
    [0.7344,         0,         0],
    [0.7188,         0,         0],
    [0.7031,         0,         0],
    [0.6875,         0,         0],
    [0.6719,         0,         0],
    [0.6562,         0,         0],
    [0.6406,         0,         0],
    [0.6250,         0,         0],
    [0.6094,         0,         0],
    [0.5938,         0,         0],
    [0.5781,         0,         0],
    [0.5625,         0,         0],
    [0.5469,         0,         0],
    [0.5312,         0,         0],
    [0.5156,         0,         0],
    [0.5000,         0,         0],
]

# On définit ensuite les variables globales
longueur_avion = 44
largeur_avion = 8
Puissance_dBm = np.zeros((int(largeur_avion / resol_y), int(longueur_avion / resol_x))) # Matrice qui récupère les valeurs depuis les fichiers textes
Debit_binaire = np.zeros((int(largeur_avion / resol_y), int(longueur_avion / resol_x)))


def dessin_carres(ax1, ax2): # Dessine les carrés de couleur noire
    x1 = np.arange(44, 60, 0.1)
    x2 = np.arange(-40, 0.05, 0.1)
    ax1.fill_between(x1, -40, 40, color='xkcd:black')
    ax2.fill_between(x1, -40, 40, color='xkcd:black')
    ax1.fill_between(x2, -40, 40, color='xkcd:black')
    ax2.fill_between(x2, -40, 40, color='xkcd:black')
    for indice,cle in enumerate(murs):
        if indice <= 19:     # On considère que les murs du nez et de la carcasse
            if murs[cle][0][0] < murs[cle][1][0]+0.1:
                x = np.arange(murs[cle][0][0], murs[cle][1][0]+0.1, 0.1)
            else:
                x = np.arange(murs[cle][1][0], murs[cle][0][0]+0.1, 0.1)
            if murs[cle][1][0] != murs[cle][0][0]:
                mu = (murs[cle][1][1]-murs[cle][0][1])/(murs[cle][1][0]-murs[cle][0][0])
                ord_or = murs[cle][1][1]-(mu*murs[cle][1][0])
                y = mu*x+ord_or
                if 3 <= indice < 12:
                    ax1.fill_between(x, -40, y, color='xkcd:black')
                    ax2.fill_between(x, -40, y, color='xkcd:black')
                if 3 > indice:
                    ax1.fill_between(x, 40, y, color='xkcd:black')
                    ax2.fill_between(x, 40, y, color='xkcd:black')
                if indice >= 12:
                    ax1.fill_between(x, 40, y, color='xkcd:black')
                    ax2.fill_between(x, 40, y, color='xkcd:black')


def dessine(): # Dessine l'avion, les émetteurs, les carrés de couleur, la colorbar
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    max_dBm = np.amax(Puissance_dBm)
    min_dBm = np.amin(Puissance_dBm)
    max_mbs = np.amax(Debit_binaire)
    min_mbs = np.amin(Debit_binaire)
    ligne, col = Puissance_dBm.shape

    # On trace les murs :
    for cle in murs:
        ax1.plot([murs[cle][0][0],murs[cle][1][0]],[murs[cle][0][1],murs[cle][1][1]],'grey')
        ax2.plot([murs[cle][0][0], murs[cle][1][0]], [murs[cle][0][1], murs[cle][1][1]], 'grey')

    # On trace les émetteurs :
    for TX in Emetteur:
        ax1.plot(TX[0], TX[1], marker="*", color='b', label='Emetteur')
        ax2.plot(TX[0], TX[1], marker="*", color='b', label='Emetteur')

    # On itère sur chaque élément d'une ligne
    for i in np.arange(0, ligne, 1):
        # print('i = ',i)
        Puissance_ligne = Puissance_dBm[i, :] # On récupère la ligne de haut en bas de l'avion
        # print('Puissance ligne = ',Puissance_ligne)
        Debit_ligne = Debit_binaire[i, :] # On récupère la ligne de haut en bas de l'avion
        # print('débit ligne = ',Debit_binaire_matrix)
        for j in np.arange(0,len(Puissance_ligne),1):
            ind_color = int((len(colors_rgb) * (Puissance_ligne[j] - min_dBm)) / (max_dBm - min_dBm)) # 256 = min/bleu, 0 = max/rouge
            if ind_color == 256:
                ind_color = 255
            # print('[i,j] = ',[i,j])
            ax1.add_patch(patches.Rectangle((j*resol_x,largeur_avion-(i+1)*resol_y), resol_x, resol_y, edgecolor=None, facecolor=colors_rgb[ind_color], fill=True))
        for j in np.arange(0,len(Puissance_ligne),1):
            ind_color = int((len(colors_rgb) * (Debit_ligne[j] - min_mbs)) / (max_mbs - min_mbs))
            if ind_color == 256:
                ind_color = 255
            ax2.add_patch(patches.Rectangle((j*resol_x,largeur_avion-(i+1)*resol_y), resol_x, resol_y, edgecolor=None, facecolor=colors_rgb[ind_color], fill=True))

    ax1.set_title("Puissance en dBm")
    ax2.set_title("Débit en Mb/s")
    ax1.set_xlim(-2.5, 46.5)
    ax1.set_ylim(-14, 22)
    ax2.set_xlim(-2.5, 46.5)
    ax2.set_ylim(-14, 22)
    dessin_carres(ax1, ax2)
    # ax1.set_facecolor('xkcd:black')
    # ax2.set_facecolor('xkcd:black')

    cmap = plt.get_cmap('jet')
    norm1 = mpl.colors.Normalize(vmin=min_dBm, vmax=max_dBm)
    norm2 = mpl.colors.Normalize(vmin=min_mbs, vmax=max_mbs)
    sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=norm1)
    sm1.set_array([])
    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm2)
    sm2.set_array([])
    plt.colorbar(sm1, ax=ax1, ticks=np.linspace(min_dBm, max_dBm, 7))
    plt.colorbar(sm2, ax=ax2, ticks=np.linspace(min_mbs, max_mbs, 7))
    plt.show()


def main(): # En fonction du nombre de rebond(s) précisé(s), ouvre un fichier texte et consigne les valeurs dans les matrices
    if nbr_rebond == 0:
        k = 0
        j = 0
        for ligne in open('P_RX_dBm-0_thread_3TX'):
            if k<int(largeur_avion/resol_y):
                if float(ligne) > -53: # Impose les limites sur la puissance en dBm
                    ligne = -53
                if float(ligne) < -78:
                    ligne = -78
                Puissance_dBm[k, j] = ligne
                j += 1
                if j%int(longueur_avion/resol_x) == 0:
                    j = 0
                    k += 1
        k = 0
        j = 0
        for ligne in open('Debit_bin-0_thread_3TX'):
            if k < int(largeur_avion / resol_y):
                if float(ligne) > 4620: # Impose les limites sur le débit binaire
                    ligne = 4620
                if float(ligne) < 27.5:
                    ligne = 27.5
                Debit_binaire[k, j] = ligne
                j += 1
                if j%int(longueur_avion/resol_x) == 0:
                    j = 0
                    k += 1

    if nbr_rebond == 1:
        k = 0
        j = 0
        for ligne in open('P_RX_dBm-1_thread_3TX'):
            if k < int(largeur_avion / resol_y):
                if float(ligne) > -53:
                    ligne = -53
                if float(ligne) < -78:
                    ligne = -78
                Puissance_dBm[k, j] = ligne
                j += 1
                if j % int(longueur_avion / resol_x) == 0:
                    j = 0
                    k += 1
        k = 0
        j = 0
        for ligne in open('Debit_bin-1_thread_3TX'):
            if k < int(largeur_avion / resol_y):
                if float(ligne) > 4620:
                    ligne = 4620
                if float(ligne) < 27.5:
                    ligne = 27.5
                Debit_binaire[k, j] = ligne
                j += 1
                if j % int(longueur_avion / resol_x) == 0:
                    j = 0
                    k += 1

    if nbr_rebond == 2:
        k = 0
        j = 0
        for ligne in open('P_RX_dBm-2_thread_mat'):
            if k < int(largeur_avion / resol_y):
                if float(ligne) > -53:
                    ligne = -53
                if float(ligne) < -78:
                    ligne = -78
                Puissance_dBm[k, j] = ligne
                j += 1
                if j % int(longueur_avion / resol_x) == 0:
                    j = 0
                    k += 1
        k = 0
        j = 0
        for ligne in open('Debit_bin-2_thread_mat'):
            if k < int(largeur_avion / resol_y):
                if float(ligne) > 4620:
                    ligne = 4620
                if float(ligne) < 27.5:
                    ligne = 27.5
                Debit_binaire[k, j] = ligne
                j += 1
                if j % int(longueur_avion / resol_x) == 0:
                    j = 0
                    k += 1
    if nbr_rebond == 3:
        k = 0
        j = 0
        for ligne in open('P_RX_dBm-3_thread_mat'):
            if k < float(largeur_avion / resol_y):
                if float(ligne) > -53:
                    ligne = -53
                if float(ligne) < -78:
                    ligne = -78
                Puissance_dBm[k, j] = ligne
                j += 1
                if j % int(longueur_avion / resol_x) == 0:
                    j = 0
                    k += 1
        k = 0
        j = 0
        for ligne in open('Debit_bin-3_thread_mat'):
            if k < int(largeur_avion / resol_y):
                if float(ligne) > 4620:
                    ligne = 4620
                if float(ligne) < 27.5:
                    ligne = 27.5
                Debit_binaire[k, j] = ligne
                j += 1
                if j % int(longueur_avion / resol_x) == 0:
                    j = 0
                    k += 1
    dessine()
main()