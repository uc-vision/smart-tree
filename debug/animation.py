from manim import *


class CreateCircle(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        tree_polygon_xyz = [
            [4, 1, 0],  # middle right
            [4, -2.5, 0],  # bottom right
            [0, -2.5, 0],  # bottom left
            [0, 3, 0],  # top left
            [2, 1, 0],  # middle
            [4, 3, 0],  # top right
        ]

        colorList = [RED, GREEN, BLUE, YELLOW]
        for i in range(200):
            point = Point(
                location=[
                    0.63 * np.random.randint(-4, 4),
                    0.37 * np.random.randint(-4, 4),
                    0,
                ],
                color=np.random.choice(colorList),
            )
            self.add(point)
        for i in range(200):
            point = Point(
                location=[
                    0.37 * np.random.randint(-4, 4),
                    0.63 * np.random.randint(-4, 4),
                    0,
                ],
                color=np.random.choice(colorList),
            )
            self.add(point)
        self.add(point)
