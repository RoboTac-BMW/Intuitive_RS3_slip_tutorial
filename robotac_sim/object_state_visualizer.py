import numpy as np
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import platform


class ObjectStateVisualiser:

    def __init__(self, env):
        # store reference to environment
        self.env = env
        self.slip_occurred = False
        self.start_detecting = False
        # create QT Application
        self.app = QtGui.QApplication([])

        # configure window
        self.win = pg.GraphicsLayoutWidget(show=True, )
        self.win.resize(720, 480)
        self.win.setWindowTitle('Visualisation')

        # enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        # create plots and curves for forces
        self.pl_fx = self.win.addPlot(title="Force (X)")
        self.pl_fy = self.win.addPlot(title="Force (Y)")
        self.pl_fz = self.win.addPlot(title="Force (Z)")
        self.pl_slip = self.win.addPlot(title="Slip Detection")

        self.win.nextRow()

        # create plots and curves for pose and velocity
        self.pl_pos_z = self.win.addPlot(title="Linear Position in Z")
        self.pl_orn_x = self.win.addPlot(title="Angular Position in X")
        self.pl_vel_z = self.win.addPlot(title="Linear Velocity in Z")
        self.pl_omega_x = self.win.addPlot(title="Angular Velocity in X")


        # Add plotters
        self.fx = self.pl_fx.plot(pen='y')
        self.fy = self.pl_fy.plot(pen='y')
        self.fz = self.pl_fz.plot(pen='y')
        self.slip = self.pl_slip.plot(pen='b')

        self.pos_z = self.pl_pos_z.plot(pen='r')
        self.orn_x = self.pl_orn_x.plot(pen='g')
        self.vel_z = self.pl_vel_z.plot(pen='r')
        self.omega_x = self.pl_omega_x.plot(pen='g')


        # buffers for plotted data
        self.b_fx = []
        self.b_fy = []
        self.b_fz = []
        self.b_slip = []

        self.b_pos_z = []
        self.b_orn_x = []
        self.b_vel_z = []
        self.b_omega_x = []

    def reset(self):
        """
        self.pl_fx.removeItem(self.threshold_line_max)
        self.pl_fx.removeItem(self.threshold_line_min)
        self.pl_fy.removeItem(self.threshold_line_max)
        self.pl_fy.removeItem(self.threshold_line_min)

        self.pl_fx.addItem(self.threshold_line_max)
        self.pl_fx.addItem(self.threshold_line_min)
        self.pl_fy.addItem(self.threshold_line_max)
        self.pl_fy.addItem(self.threshold_line_min)
        """
        self.win.close()

    def update_plot(self):

        # store new data
        self.b_fx.append(self.env.tactile_sensor.force_x)
        self.b_fy.append(self.env.tactile_sensor.force_y)
        self.b_fz.append(self.env.tactile_sensor.force_z)

        pos, orn, lin_vel, ang_vel = self.env.object.get_observation()

        self.b_pos_z.append(pos[2])
        self.b_orn_x.append(orn[0])
        self.b_vel_z.append(lin_vel[2])
        self.b_omega_x.append(ang_vel[0])

        if self.start_detecting:
            print("Checking slip ", lin_vel, ang_vel)
            if lin_vel[2] < -0.01 or abs(orn[0]) > 0.02:

                self.b_slip.append(1)
                self.slip_occurred = True
            else:
                if not self.slip_occurred:
                    self.b_slip.append(0)
                else:
                    self.b_slip.append(1)
        else:
            self.b_slip.append(0)
        # plot new data
        self.fx.setData(self.b_fx)
        self.fy.setData(self.b_fy)
        self.fz.setData(self.b_fz)
        self.slip.setData(self.b_slip)

        self.pos_z.setData(self.b_pos_z)
        self.orn_x.setData(self.b_orn_x)
        self.vel_z.setData(self.b_vel_z)
        self.omega_x.setData(self.b_omega_x)


        # on macOS, calling processEvents() is unnecessary
        # and even results in an error. only do so on Linux
        if platform.system() == 'Linux' or platform.system() == 'Windows':
            self.app.processEvents()