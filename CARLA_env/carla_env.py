from CARLA_env import *
from CARLA_env.carla_data_provider import CarlaDataProvider


def norm(vector):
    return np.linalg.norm(vector)


def get_vehicle_info(vehicle):
    angular_velocity = vehicle.get_angular_velocity()
    acceleration = vehicle.get_acceleration()
    velocity = vehicle.get_velocity()
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation
    vehicle_info = [
        location.x,
        location.y,
        location.z,
        rotation.pitch,
        rotation.roll,
        rotation.yaw,
        velocity.x,
        velocity.y,
        velocity.z,
        angular_velocity.x,
        angular_velocity.y,
        angular_velocity.z,
        acceleration.x,
        acceleration.y,
        acceleration.z,
    ]
    return vehicle_info


def get_vehicle_vel(vehicle):
    velocity = vehicle.get_velocity()
    return norm(np.array([velocity.x, velocity.y]))


def lidar_callback(lidar_data, sensor_queue):
    # lidar_data.save_to_disk(os.path.join('./outputs/output_synchronized', '%06d.ply' % lidar_data.frame))
    sensor_queue.put(lidar_data.frame)


def image_callback(image_data, sensor_queue):
    # image_data.save_to_disk(os.path.join('./outputs/output_synchronized', '%06d.png' % image_data.frame))
    sensor_queue.put(image_data.frame)


class CarlaEnv(object):
    def __init__(self, max_step):
        self.max_step = max_step
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town02')
        self.original_settings = self.world.get_settings()
        self.blueprint_library = self.world.get_blueprint_library()

        self.cur_step = 0
        self.actor_list = []
        self.sensor_list = []
        self.image_deque = Queue()
        self.lidar_deque = Queue()
        self.ego_vehicle = None
        self.pos_log = []
        self.action_space = np.array([[0.75, 0, 0],
                                      [0, 0, 0.75],
                                      [0.5, 0.5, 0],
                                      [0.5, -0.5, 0]])

        CarlaDataProvider.set_client(self.client)

    def init(self):
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        ego_vehicle_bp = self.blueprint_library.find('vehicle.dodge.charger_police_2020')
        transform = self.world.get_map().get_spawn_points()[67]
        self.ego_vehicle = self.world.spawn_actor(ego_vehicle_bp, transform)

        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.register_actor(self.ego_vehicle)

        self.actor_list.append(self.ego_vehicle)

        camera_bp = self.blueprint_library.find('sensor.camera.rgb')

        camera_location = carla.Location(1.5, 0, 2.4)
        camera_rotation = carla.Rotation(0, 0, 0)
        camera_transform = carla.Transform(camera_location, camera_rotation)

        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        camera.listen(lambda image: image_callback(image, self.image_deque))
        self.sensor_list.append(camera)

        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', str(32))
        lidar_bp.set_attribute('points_per_second', str(90000))
        lidar_bp.set_attribute('rotation_frequency', str(40))
        lidar_bp.set_attribute('range', str(20))

        lidar_location = carla.Location(0, 0, 2)
        lidar_rotation = carla.Rotation(0, 0, 0)
        lidar_transform = carla.Transform(lidar_location, lidar_rotation)

        lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_vehicle)
        lidar.listen(lambda point_cloud: lidar_callback(point_cloud, self.lidar_deque))
        self.sensor_list.append(lidar)
        init_step = 10
        for i in range(init_step):
            self.step(4)
        return self._state(), self._info(), False

    def reset(self):
        self.world.apply_settings(self.original_settings)
        # print('destroying actors')
        for sensor in self.sensor_list:
            sensor.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])

        self.cur_step = 0
        self.actor_list = []
        self.sensor_list = []
        self.image_deque = Queue()
        self.lidar_deque = Queue()
        self.ego_vehicle = None
        # print('done.')

    def step(self, action_index):
        if action_index == 4:
            action = [0., 0., 0.]
        else:
            action = self.action_space[action_index]
        control = carla.VehicleControl()
        control.throttle = action[0]
        control.steer = action[1]
        control.brake = action[2]
        self.ego_vehicle.apply_control(control)
        self.world.tick()
        self.cur_step += 1
        spectator = self.world.get_spectator()
        transform = self.ego_vehicle.get_transform()

        spectator_location = transform.location + carla.Location(0, 0, 40)
        spectator_rotation = carla.Rotation(-90, 0, 0)
        spectator_transform = carla.Transform(spectator_location, spectator_rotation)
        spectator.set_transform(spectator_transform)

        CarlaDataProvider.on_carla_tick()
        print('-' * 50)
        print(CarlaDataProvider.get_location(self.ego_vehicle))

        return self._state(), self._reward(action), self._done(), self._info()

    def _reward(self, action):
        if action[0] > 0.5:
            return 5
        else:
            return -1

    def _state(self):
        transform = self.ego_vehicle.get_transform()
        return [transform.location.x, transform.location.y, transform.rotation.yaw]

    def _done(self):
        if self.cur_step > self.max_step:
            return True
        else:
            return False

    def _info(self):
        return get_vehicle_info(self.ego_vehicle)
