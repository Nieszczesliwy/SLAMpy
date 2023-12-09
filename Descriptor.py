 
from multiprocessing import Process, Queue
import numpy as np
import OpenGL.GL as gl
import pangolin
import g2o

class WorldPoint:
    def __init__(self, world, location):
        self.position = location
        self.observed_frames = []
        self.observation_indices = []

        self.point_id = len(world.world_points)
        world.world_points.append(self)

    def add_observation(self, frame, index):
        frame.tracked_points[index] = self
        self.observed_frames.append(frame)
        self.observation_indices.append(index)

class WorldDescriptor:
    def __init__(self):
        self.frames = []
        self.world_points = []
        self.optimization_state = None
        self.queue = None

    def optimize(self):
        optimization_error = optimize(self.frames, self.world_points, local_window, fix_points, verbose, rounds)

        removed_points_count = 0
        for point in self.world_points:
            old_point = len(point.observed_frames) <= 4 and point.observed_frames[-1].frame_id + 7 < self.max_frame
            projection_errors = []
            for frame, index in zip(point.observed_frames, point.observation_indices):
                uv = frame.keypoints[index]
                projected = np.dot(frame.frame_pose[:3], point.to_homogeneous())
                projected = projected[0:2] / projected[2]
                projection_errors.append(np.linalg.norm(projected - uv))
            if old_point or np.mean(projection_errors) > CULLING_ERR_THRES:
                removed_points_count += 1
                self.world_points.remove(point)
                point.delete()

        return optimization_error

    def create_viewer(self):
        self.queue = Queue()
        self.viewer_process = Process(target=self.viewer_thread, args=(self.queue,))
        self.viewer_process.daemon = True
        self.viewer_process.start()

    def viewer_thread(self, queue):
        self.initialize_viewer(1024, 768)
        while True:
            self.refresh_viewer(queue)

    def add_observation(self, frame, index):
        if index < len(frame.tracked_points):
            frame.tracked_points[index] = self
        else:
            print(f"Attempted to add observation with index {index}, which is out of range.")


    def initialize_viewer(self, width, height):
        pangolin.CreateWindowAndBind('Main', width, height)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scene_camera = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(width, height, 420, 420, width // 2, height // 2, 0.2, 10000),
            pangolin.ModelViewLookAt(0, -10, -8, 0, 0, 0, 0, -1, 0))
        self.viewer_handler = pangolin.Handler3D(self.scene_camera)

        self.display_camera = pangolin.CreateDisplay()
        self.display_camera.SetBounds(0.0, 1.0, 0.0, 1.0, -width / height)
        self.display_camera.SetHandler(self.viewer_handler)

    def refresh_viewer(self, queue):
        if self.optimization_state is None or not queue.empty():
            self.optimization_state = queue.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0, 0, 0, 0)
        self.display_camera.Activate(self.scene_camera)

        gl.glPointSize(2)
        gl.glColor3f(0.184314, 0.309804, 0.184314)
        pangolin.DrawPoints(self.optimization_state[1] + 1)
        gl.glPointSize(1)
        gl.glColor3f(0.3099, 0.3099, 0.184314)
        pangolin.DrawPoints(self.optimization_state[1])

        gl.glColor3f(0.0, 1.0, 1.0)
        pangolin.DrawCameras(self.optimization_state[0])

        pangolin.FinishFrame()

    def update_display(self):
        if self.queue is None:
            return
        frame_poses, point_positions = [], []
        for frame in self.frames:
            frame_poses.append(frame.camera_pose)
        for point in self.world_points:
            point_positions.append(point.position)
        self.queue.put((np.array(frame_poses), np.array(point_positions)))
