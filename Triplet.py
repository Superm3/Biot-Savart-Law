import math;
import numpy as np;

class triplet:
    def __init__(self, x, y ,z):
        self.x = x; self.y = y; self.z = z;

    def dot(self, other):
        return ((self.x * other.x) + (self.y * other.y) + (self.z * other.z));

    def cross(self, other):
        new_x = (self.y * other.z) - (self.z * other.y);
        new_y = -1 * ((self.x * other.z) - (self.z * other.x));
        new_z = (self.x * other.y) - (self.y * other.x);

        return triplet(new_x, new_y, new_z);

    def norm_sq(self):
        return self.dot(self);

    def norm(self):
        return np.sqrt(self.norm_sq());

    def add(self, other):
        return triplet(self.x + other.x, self.y + other.y, self.z + other.z);

    def sub(self, other):
        return triplet(self.x - other.x, self.y - other.y, self.z - other.z);

    def distance_from(self, other):
        return other.sub(self);

    def scale(self, scalar):
        return triplet(self.x * scalar, self.y * scalar, self.z * scalar);

    def unit(self):
        return self.scale(1.0 / self.norm());

    def __str__(self):
        return str(self.x) + "\t" + str(self.y) + "\t" + str(self.z);

    def rotate_x(self, theta):
        new_x = self.x;
        new_y = (self.y * np.cos(theta)) - (self.z * np.sin(theta));
        new_z = (self.y * np.sin(theta)) + (self.z * np.cos(theta));
        return triplet(new_x, new_y, new_z);

    def rotate_y(self, theta):
        new_x = self.x * np.cos(theta) + self.z * np.sin(theta);
        new_y = self.y;
        new_z = -1.0 * self.x * np.sin(theta) + self.z * np.cos(theta);
        return triplet(new_x, new_y, new_z);

    def rotate_z(self, theta):
        new_x = self.x * np.cos(theta) - self.y * np.sin(theta);
        new_y = self.x * np.sin(theta) + self.y * np.cos(theta);
        new_z = self.z;
        return triplet(new_x, new_y, new_z);
