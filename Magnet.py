import numpy as np;
import triplet as tp;
import math;
import matplotlib.pyplot as plt;

MU_0 =  4 * np.pi * math.pow(10, -7);
steps = 1000;

class line:
    def __init__(self, end1, end2, I):
        self.end1 = end1; self.end2 = end2; self.I = I;


    def value_at(self, point):
        B = tp.triplet(0, 0, 0);
        delta = self.end2.sub(self.end1).scale(1 / steps);
        q_left = self.end1.scale(1);

        while((q_left.x < self.end2.x) | (q_left.y < self.end2.y) | (q_left.z < self.end2.z)):
            q_right = q_left.add(delta);
            q_mid = q_left.add(q_right).scale(1.0 / 2.0);
            r_sq = point.sub(q_mid).norm_sq();
            r_unit = point.sub(q_mid).unit();

            B = B.add((delta.cross(r_unit)).scale(1.0 / r_sq));
            q_left = q_left.add(delta);

        return B.scale(MU_0 * self.I / (np.pi * 4.0));

class circle:
    def __init__(self, rad, I):
        self.rad = rad; self.I = I;
        self.circ = 2.0 * np.pi * rad;

    def value_at(self, point):
        B = tp.triplet(0, 0, 0);
        delta_mag = self.circ / steps;

        for t in np.arange(steps):
            x_pos = self.rad * np.cos(2.0 * np.pi * t / steps);
            y_pos = self.rad * np.sin(2.0 * np.pi * t / steps);
            z_pos = 0.0;
            q = tp.triplet(x_pos, y_pos, z_pos);

            dx = -1.0 * np.sin(2.0 * np.pi * t / steps);
            dy = np.cos(2.0 * np.pi * t / steps);
            dz = 0.0;
            delta = (tp.triplet(dx, dy, dz)).unit().scale(delta_mag);

            r_sq = point.sub(q).norm_sq();
            r_unit = point.sub(q).unit();

            B = B.add((delta.cross(r_unit)).scale(1.0 / r_sq));

        return B.scale(self.I * MU_0 / (np.pi * 4.0));

class cylinder:
    def __init__(self, rad, height, loops, I):
        self.rad = rad; self.height = height; self.loops = loops; self.I = I;

    def value_at(self, point):

        B = tp.triplet(0, 0, 0);
        circ = circle(self.rad, self.I);
        for i in np.linspace(-self.height / 2, self.height / 2, self.loops):
            B = B.add(circ.value_at(point.add(tp.triplet(0, 0, -i))));

        return B;

class rectangle:
    def __init__(self, x_len, y_len, I):
        self.corner1 = tp.triplet(- x_len / 2, - y_len / 2, 0);
        self.corner2 = tp.triplet(x_len / 2, - y_len / 2, 0);
        self.corner3 = tp.triplet(x_len / 2, y_len / 2, 0);
        self.corner4 = tp.triplet(- x_len / 2, y_len / 2, 0);
        self.I = I;

    def value_at(self, point):
        line1 = line(self.corner1, self.corner2, self.I);
        line2 = line(self.corner2, self.corner3, self.I);
        line3 = line(self.corner4, self.corner3, self.I);
        line4 = line(self.corner1, self.corner4, self.I);

        b1 = line1.value_at(point);
        b2 = line2.value_at(point);
        b3 = line3.value_at(point).scale(-1.0);
        b4 = line4.value_at(point).scale(-1.0);

        return b1.add(b2).add(b3).add(b4);

class box:
    def __init__(self, x_len, y_len, z_len, loops, I):
        self.x_len = x_len; self.y_len = y_len; self.z_len = z_len;
        self.loops = loops; self.I = I;

    def value_at(self, point):

        B = tp.triplet(0, 0, 0);
        rec = rectangle(self.x_len, self.y_len, self.I);
        for i in np.linspace(-self.z_len / 2, self.z_len / 2, self.loops):
            B = B.add(rec.value_at(point.add(tp.triplet(0, 0, -i))));

        return B;
