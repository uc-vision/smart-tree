from functools import partial
import numpy as np
from tqdm import tqdm

from torch.multiprocessing import Pool
from smart_tree.util.misc import flatten_list
from smart_tree.util.visualizer.camera import Renderer

from multiprocessing import Process, Queue
import wandb


def render_cloud(
    renderer, labelled_cloud, camera_position=[1, 0, 0], camera_up=[0, 1, 0]
):
    segmented_img = renderer.capture(
        [labelled_cloud.geoms["segmented_cld"]],
        camera_position,
        camera_up,
    )

    projected_img = renderer.capture(
        [labelled_cloud.geoms["projected"]],
        camera_position,
        camera_up,
    )

    return [np.asarray(segmented_img), np.asarray(projected_img)]


def log_images(wandb_run, name, images, epoch):
    wandb_run.log({f"{name}": [wandb.Image(img) for img in images]})


class RenderQueue:
    def __init__(self, wandb_run, image_size=(960, 540)):
        self.queue = Queue(2)

        self.image_size = image_size
        self.wandb_run = wandb_run

        self.worker = Process(target=self.render_worker)
        self.worker.start()

    def render_worker(self):
        renderer = Renderer(*self.image_size)
        item = self.queue.get()

        while item is not None:
            clouds, epoch = item

            renders = [render_cloud(renderer, cloud) for cloud in clouds]
            log_images(self.wandb_run, "Test Output", flatten_list(renders), epoch)

            item = self.queue.get()

    def render_clouds(self, clouds, epoch):
        self.queue.put((clouds, epoch))

    def join(self):
        self.queue.put(None)
        self.worker.join()
