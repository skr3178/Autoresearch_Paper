#!/usr/bin/env python3
"""Debug script to check Agent object structure."""

import sys
sys.path.insert(0, '/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit')

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# Dataset paths
data_root = '/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-extracted/data/cache/mini'
map_root = '/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/maps/nuplan-maps-v1.0'
map_version = '1.0'

# Create scenario builder
builder = NuPlanScenarioBuilder(
    data_root=data_root,
    map_root=map_root,
    sensor_root='',
    db_files=None,
    map_version=map_version,
    include_cameras=False,
    max_workers=1,
    verbose=True,
    vehicle_parameters=get_pacifica_parameters()
)

# Load one scenario
scenario_filter = ScenarioFilter(
    scenario_tokens=None,
    log_names=None,
    scenario_types=None,
    num_scenarios_per_type=1,
    limit_total_scenarios=1,
    expand_scenarios=False,
    remove_invalid_goals=True,
    shuffle=False,
    ego_displacement_minimum_m=None,
    ego_start_speed_threshold=None,
    ego_stop_speed_threshold=None,
    speed_noise_tolerance=None,
    ego_route_radius=None,
    map_names=None,
    timestamp_threshold_s=None,
    token_set_path=None,
    fraction_in_token_set_threshold=None
)

worker = Sequential()
scenarios = builder.get_scenarios(scenario_filter, worker)
scenario = scenarios[0]

# Get tracked objects
iteration = 0
gt_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=None)
tracked_objects = scenario.get_tracked_objects_at_iteration(iteration, gt_sampling)
objects = tracked_objects.tracked_objects.tracked_objects

print(f"Number of objects: {len(objects)}")
if len(objects) > 0:
    obj = objects[0]
    print(f"\nFirst object type: {type(obj)}")
    print(f"Object class name: {obj.__class__.__name__}")
    print(f"\nObject attributes:")
    for attr in dir(obj):
        if not attr.startswith('_'):
            print(f"  {attr}")
    
    print(f"\nChecking object structure:")
    print(f"  hasattr(obj, 'oriented_box'): {hasattr(obj, 'oriented_box')}")
    print(f"  hasattr(obj, 'box'): {hasattr(obj, 'box')}")
    
    if hasattr(obj, 'box'):
        print(f"\n  obj.box type: {type(obj.box)}")
        print(f"  obj.box.center: {obj.box.center}")
        print(f"  obj.box.center.x: {obj.box.center.x}")
        print(f"  obj.box.center.y: {obj.box.center.y}")
        print(f"  obj.box.center.heading: {obj.box.center.heading}")
