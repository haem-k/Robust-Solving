import numpy as np

from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate

file = open("model.vs")
vert = file.read()
file.close()

file = open("model.fs")
frag = file.read()
file.close()


import lbs_import as lbs
import local_reference
import marker_config as mc

# Marker class array
test_markers = lbs.markers

# nof x noj x row(4) x col(4)
test_motion = lbs.motion

# 4*4 matrix for each index
test_offsets = lbs.offsets

# load torso marker name, and rigidBody from txt file
torso, test_rigidBody = local_reference.loadRigidBody()

# calculate marker configuration and get sampled Z
# Z = mc.getMarkerConfig(test_motion, test_offsets, test_markers)
# sampledZ = mc.sampleMarkerConfig(Z)

# number of frame, number of joint, number of markers
test_nof = np.size(test_motion, 0)  # frame number
test_noj = np.size(test_motion, 1)  # joint number
test_nom = np.size(test_markers, 0) # marker number

# ------------------------------------------------------------ Canvas class ---
class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600))
        ps = self.pixel_scale

        # Create vertices
        n = 100
        data = np.zeros(n, [('a_id', np.float32)])
        for i in range(n):
            data[i][0] = np.float32(i)

        self.frame = 0
        self.x = 0
        self.y = 0
        self.z = 5
        self.program = gloo.Program(vert, frag)
        
        self.view = translate((self.x, self.y, -self.z))
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.apply_zoom()

        self.program.bind(gloo.VertexBuffer(data))
        
        # initialize uniform
        self.program['u_model'] = self.model
        self.program['u_view']  = self.view
        self.program['u_size']  = 5 / self.z
        self.program['u_rad'] = np.float32(1.0)
        for i in range(n):
            self.program[f'u_pos[{i}]'] = [0, 0, 0]
        
        self.theta = 0
        self.phi = 0

        gloo.set_state('translucent', clear_color='white')

        self.timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.run = True

        self.show()
        self.slowDown = 0

        self.init_pose = np.eye(4, 4)

    def on_key_press(self, event):
        if event.text == ' ':
            # if self.timer.running:
            #     self.timer.stop()
            # else:
            #     self.timer.start()
            self.run = not self.run

    def on_timer(self, event):
        self.program['u_model'] = self.model
        
        if self.run:
            if self.slowDown > 2:
                self.frame = self.frame + 1
                if self.frame >= test_nof:
                    self.frame = 0
                self.slowDown = 0
            else :
                self.slowDown = self.slowDown + 1
        
        self.update()

    def on_resize(self, event):
        self.apply_zoom()

    def on_mouse_wheel(self, event):
        self.z -= event.delta[1]
        self.z = max(2, self.z)
        self.view = translate((self.x, self.y, -self.z))

        self.program['u_view'] = self.view
        self.program['u_size'] = 5 / self.z
        #self.update()

    def on_mouse_move(self, event):
        """Pan the view based on the change in mouse position."""
        if event.is_dragging and event.buttons[0] == 2:
            x0, y0 = event.last_event.pos[0], event.last_event.pos[1]
            x1, y1 = event.pos[0], event.pos[1]
            self.x = self.x + 0.001 * (x1 - x0)
            self.y = self.y - 0.001 * (y1 - y0)
            
            self.view = translate((self.x, self.y, -self.z))
            self.program['u_view'] = self.view
        elif event.is_dragging and event.buttons[0] == 1:
            x0, y0 = event.last_event.pos[0], event.last_event.pos[1]
            x1, y1 = event.pos[0], event.pos[1]

            #self.theta += (y1 - y0)
            self.phi += (x1 - x0)
            self.model = np.dot(rotate(self.theta, (1, 0, 0)),
                                rotate(self.phi, (0, 1, 0)))
            # self.translate_center(X1 - X0, Y1 - Y0)

    def apply_zoom(self):
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 1.0, 1000.0)
        self.program['u_projection'] = self.projection


    # Called evey frame
    def on_draw(self, event):
        gloo.clear()

        # drawing scale parameter
        scale = 0.01

        ##################### Draw Corrupted Markers #####################
        # self.program['u_color'] = [1, 0, 0] # color (RGB)
        # cnt = 0
        # corruptedMarkers = []

        # # For all markers,
        # for marker in test_markers:
        #     # corrupt a marker
        #     corruptedMarker = lbs.corruptMarkers(marker, 0.1, 0.1, 1)
        #     corruptedMarkers.append(corruptedMarker)
        #     pos = corruptedMarker.globalPosition(self.frame, test_offsets, test_motion)
        #     pos = np.squeeze(pos)

        #     uniform_name = f'u_pos[{cnt}]'
        #     self.program[uniform_name] = [scale * pos[0], scale * pos[1], scale * pos[2]]
        #     cnt = cnt + 1

        # # destination markers for rigid body
        # destination = np.zeros((len(torso), 4))
        # # For all markers in the torso,
        # for j in range(len(torso)):
        #     # Get a marker from the torso
        #     markerName = torso[j]
        #     m = [m for m in corruptedMarkers if markerName in m.name][0]
        #     pos = m.globalPosition(self.frame, test_offsets, test_motion)
        #     pos = np.squeeze(pos)
        #     destination[j] = pos
        # destination[:,3] = 1

        # # Compute local reference frame with icp algorithm
        # # Tranformation based on corrupted markers
        # F, distances, iteration = localRef.icp(test_rigidBody, destination, init_pose=self.init_pose, max_iterations=100, tolerance=0.001)
        # print("frame number: " + str(self.frame) + "\nmean error: " + str(np.mean(distances)) + "\niteration: " + str(iteration) + "\n")

        # self.program['u_rad'] = np.float32(1.0)
        # self.program['u_num'] = np.float32(test_nom) # nom: number of markers
        # self.program.draw('points')

        
        ##################### Draw Original Markers #####################
        # self.program['u_color'] = [1, 0, 0] # color (RGB)
        # cnt = 0
        # for marker in test_markers:
        #     pos = marker.globalPosition(self.frame, test_offsets, test_motion)
        #     pos = np.squeeze(pos)
            
        #     uniform_name = f'u_pos[{cnt}]'
        #     self.program[uniform_name] = [scale * pos[0], scale * pos[1], scale * pos[2]]
        #     cnt = cnt + 1

        # self.program['u_rad'] = np.float32(1.0)
        # self.program['u_num'] = np.float32(test_nom) # noj: number of joints
        # self.program.draw('points')
        

        ##################### Draw Joints #####################
        self.program['u_color'] = [0, 0, 0]
        for i in range(test_noj): 
            ji = lbs.getJointPos(test_motion, self.frame, i)
            uniform_name = f'u_pos[{i}]'
            self.program[uniform_name] = [scale * ji[0], scale * ji[1], scale * ji[2]]
        
        self.program['u_rad'] = np.float32(1.0)
        self.program['u_num'] = np.float32(test_noj) # noj: number of joint
        self.program.draw('points')
        

        ##################### Draw with Sampled Z #####################
        self.program['u_color'] = [0, 1, 0]
        cnt = 0
        # corruptedMarkers = []

        for m in range(test_nom):
            pos = np.zeros((4,))
            marker = test_markers[m]

            for j in range(len(marker.indices)):
                # Get effective joint index
                index = marker.indices[j]

                localOffset = np.append(sampledZ[self.frame][m][index], 1)
                jntTrf = test_motion[self.frame][index]
                pos += np.matmul(jntTrf, localOffset.T) * marker.weights[j]

            if self.frame == 1:
                print(pos)
            # corrupt a marker
            pos = lbs.corruptMarkers(pos, 0.1, 0.1, 1)
            # corruptedMarkers.append(corruptedMarker)
    
            uniform_name = f'u_pos[{cnt}]'
            self.program[uniform_name] = [scale * pos[0], scale * pos[1], scale * pos[2]]
            cnt = cnt + 1
        
        self.program['u_rad'] = np.float32(2.0)
        self.program['u_num'] = np.float32(test_nom) # nom: number of markers
        self.program.draw('points')



        ##################### Draw Rigid Body #####################
        # self.program['u_color'] = [0, 1, 0]
        # rb_num = np.size(test_rigidBody, axis=0)

        # for i in range(rb_num):
        #     rb = test_rigidBody[i]
        #     # Multiply local reference frame F to check if it is computed correctly
        #     rb = np.matmul(F, rb.T)

        #     uniform_name = f'u_pos[{i}]'
        #     self.program[uniform_name] = [scale * rb[0], scale * rb[1], scale * rb[2]]
        
        # self.program['u_rad'] = np.float32(2.0)
        # self.program['u_num'] = np.float32(rb_num) # rigid body marker number
        # self.program.draw('points')



if __name__ == '__main__':
    c = Canvas()
    #c.measure_fps()
    app.run()
