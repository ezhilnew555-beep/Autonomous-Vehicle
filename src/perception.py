r"""
src/perception.py
=================
Simulated Perception System for Autonomous Vehicle (Phase 5).

DISCLAIMER:
This is a simulated sensor/perception engine designed for simulation and testing.
It models range limitations, sensor delays, missed detections, and kinematic tracking.
It is NOT a hardware camera, LiDAR, or deep learning model.

Key Features:
- Clean separation between ground truth (`DynamicObstacle`) and perceived tracks (`PerceivedObject`).
- Configurable detection range (Euclidean distance from ego vehicle).
- Configurable detection delay (temporal latency buffer before observations manifest).
- Configurable missed detection probability (stochastic dropout).
- Persistent track ID assignment across consecutive observations.
- Kinematic velocity and speed estimation derived from successive position observations over time:
  $v_x = \frac{\Delta x}{\Delta t}, \quad v_y = \frac{\Delta y}{\Delta t}, \quad speed = \sqrt{v_x^2 + v_y^2}$
- Coasting / track-loss handling when an object exits the detection field or disappears.
"""

from collections import deque
from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Optional, Tuple
import pygame

import config
from src.obstacles import DynamicObstacle, ObstacleType


@dataclass
class Observation:
    """Raw observation timestamped at the moment of perception."""
    timestamp: float
    x: float
    y: float
    heading: float
    category: ObstacleType
    length: float
    width: float


@dataclass
class PerceivedObject:
    """
    Tracked object estimated by the simulated perception system.
    Maintains persistent track ID, estimated position, and estimated velocity.
    """
    track_id: int
    category: ObstacleType
    estimated_x: float
    estimated_y: float
    estimated_heading: float
    estimated_vx: float = 0.0
    estimated_vy: float = 0.0
    estimated_speed: float = 0.0
    length: float = 24.0
    width: float = 14.0
    last_seen_time: float = 0.0
    confidence: float = 1.0
    is_in_range: bool = True

    # Internal history of observations for finite-difference filtering
    _history: deque = field(default_factory=lambda: deque(maxlen=6), repr=False)

    def add_observation(self, obs: Observation) -> None:
        """Incorporate a new observation and re-estimate position and velocity."""
        self._history.append(obs)
        self.last_seen_time = obs.timestamp
        self.is_in_range = True
        self.category = obs.category
        self.length = obs.length
        self.width = obs.width
        self.estimated_heading = obs.heading

        # Finite differences / velocity estimation if history >= 2
        if len(self._history) >= 2:
            prev = self._history[-2]
            curr = self._history[-1]
            dt = curr.timestamp - prev.timestamp

            if dt > 1e-4:
                raw_vx = (curr.x - prev.x) / dt
                raw_vy = (curr.y - prev.y) / dt

                if len(self._history) == 2:
                    self.estimated_vx = raw_vx
                    self.estimated_vy = raw_vy
                else:
                    alpha = 0.6
                    self.estimated_vx = alpha * raw_vx + (1.0 - alpha) * self.estimated_vx
                    self.estimated_vy = alpha * raw_vy + (1.0 - alpha) * self.estimated_vy

                self.estimated_speed = math.hypot(self.estimated_vx, self.estimated_vy)

        self.estimated_x = obs.x
        self.estimated_y = obs.y


class PerceptionSystem:
    """
    Simulated perception sensor engine attached to the ego vehicle.
    """

    def __init__(
        self,
        detection_range: float = config.DEFAULT_DETECTION_RANGE,
        detection_delay: float = config.DEFAULT_DETECTION_DELAY,
        miss_rate: float = config.DEFAULT_MISS_RATE,
        track_timeout: float = 1.0,  # seconds before dropped track is pruned
    ) -> None:
        self.detection_range = detection_range
        self.detection_delay = detection_delay
        self.miss_rate = miss_rate
        self.track_timeout = track_timeout

        # Active tracks keyed by ground-truth object ID
        self.tracks: Dict[int, PerceivedObject] = {}

        # FIFO buffer of pending delayed observations: (delivery_time, obj_id, Observation)
        self._delay_buffer: deque = deque()

        self.current_sim_time: float = 0.0
        self.show_detection_range: bool = True
        self.show_labels: bool = True

    def clear(self) -> None:
        """Clear all active tracks and observation queues."""
        self.tracks.clear()
        self._delay_buffer.clear()
        self.current_sim_time = 0.0

    def update(
        self,
        sim_time: float,
        ego_x: float,
        ego_y: float,
        ground_truth_objects: List[DynamicObstacle],
    ) -> List[PerceivedObject]:
        """
        Simulate perception step for current simulation timestamp:
        1. Query ground truth objects within detection range.
        2. Apply simulated miss rate (sensor dropout).
        3. Enqueue observations with configured latency delay.
        4. Flush ready observations from delay buffer into tracks.
        5. Prune tracks that haven't been observed within timeout.
        6. Return currently perceived objects.
        """
        self.current_sim_time = sim_time
        detected_gt_ids = set()

        for obj in ground_truth_objects:
            if not obj.is_active:
                continue

            dist = math.hypot(obj.x - ego_x, obj.y - ego_y)
            if dist <= self.detection_range:
                detected_gt_ids.add(obj.id)

                # Simulated sensor dropout
                if self.miss_rate > 0.0 and random.random() < self.miss_rate:
                    continue

                # Create observation
                obs = Observation(
                    timestamp=sim_time,
                    x=obj.x,
                    y=obj.y,
                    heading=obj.heading,
                    category=obj.category,
                    length=obj.length,
                    width=obj.width,
                )

                delivery_time = sim_time + self.detection_delay
                self._delay_buffer.append((delivery_time, obj.id, obs))

        # Process mature observations whose delivery delay has elapsed
        while self._delay_buffer and self._delay_buffer[0][0] <= sim_time:
            _, obj_id, obs = self._delay_buffer.popleft()

            if obj_id not in self.tracks:
                # Initialize new persistent track
                self.tracks[obj_id] = PerceivedObject(
                    track_id=obj_id,
                    category=obs.category,
                    estimated_x=obs.x,
                    estimated_y=obs.y,
                    estimated_heading=obs.heading,
                    length=obs.length,
                    width=obs.width,
                    last_seen_time=obs.timestamp,
                )

            self.tracks[obj_id].add_observation(obs)

        # Mark tracks out-of-range or prune dead tracks
        dead_track_ids = []
        for tid, track in self.tracks.items():
            time_since_seen = sim_time - track.last_seen_time
            if tid not in detected_gt_ids:
                track.is_in_range = False

            if time_since_seen > self.track_timeout:
                dead_track_ids.append(tid)

        for tid in dead_track_ids:
            del self.tracks[tid]

        return list(self.tracks.values())

    @property
    def perceived_objects(self) -> List[PerceivedObject]:
        """Return snapshot of active perceived tracks."""
        return list(self.tracks.values())


# ---------------------------------------------------------------------------
# Perception Visualizations
# ---------------------------------------------------------------------------
def draw_perception_frustum(
    surface: pygame.Surface,
    ego_x: float,
    ego_y: float,
    detection_range: float,
) -> None:
    """Render semi-transparent circle showing simulated sensor coverage."""
    r = int(round(detection_range))
    cx, cy = int(round(ego_x)), int(round(ego_y))

    # Bounded surface for alpha blending
    frustum_surf = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
    pygame.draw.circle(frustum_surf, config.C_RANGE_CIRCLE, (r, r), r)
    pygame.draw.circle(frustum_surf, config.C_RANGE_BORDER, (r, r), r, width=1)
    surface.blit(frustum_surf, (cx - r, cy - r))


def draw_perceived_tracks(
    surface: pygame.Surface,
    tracks: List[PerceivedObject],
    show_labels: bool,
    font: pygame.font.Font,
) -> None:
    """
    Render bounding boxes, velocity vectors, and tracking tags for perceived objects.
    Visual distinction: bright cyan/neon green bounding box around confirmed detected objects.
    """
    for trk in tracks:
        x, y = trk.estimated_x, trk.estimated_y
        hl = trk.length / 2.0 + 3.0
        hw = trk.width / 2.0 + 3.0
        cos_h = math.cos(trk.estimated_heading)
        sin_h = math.sin(trk.estimated_heading)

        def rot(lx: float, ly: float) -> Tuple[int, int]:
            return int(round(x + lx * cos_h - ly * sin_h)), int(round(y + lx * sin_h + ly * cos_h))

        box_color = config.C_PERCEIVED_BBOX if trk.is_in_range else config.C_OUT_OF_RANGE_TAG
        corners = [rot(hl, -hw), rot(hl, hw), rot(-hl, hw), rot(-hl, -hw)]
        pygame.draw.polygon(surface, box_color, corners, width=2)

        # Velocity arrow vector if moving
        if trk.estimated_speed > 2.0:
            scale = 0.4
            arrow_end = (int(round(x + trk.estimated_vx * scale)), int(round(y + trk.estimated_vy * scale)))
            pygame.draw.line(surface, config.C_PERCEIVED_BBOX, (int(round(x)), int(round(y))), arrow_end, 2)
            pygame.draw.circle(surface, config.C_PERCEIVED_BBOX, arrow_end, 3)

        # Track Label
        if show_labels:
            status_str = f"#{trk.track_id} {trk.category.value} | {trk.estimated_speed:.1f}px/s"
            tag_col = config.C_PERCEIVED_LABEL if trk.is_in_range else config.C_OUT_OF_RANGE_TAG
            tag_surf = font.render(status_str, True, tag_col)
            surface.blit(tag_surf, (int(round(x - tag_surf.get_width() / 2)), int(round(y + hw + 2))))
