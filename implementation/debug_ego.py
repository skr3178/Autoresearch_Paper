#!/usr/bin/env python3
"""Debug script to check EgoState object structure."""

import sys
sys.path.insert(0, '/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-devkit')

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

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

# Get ego future trajectory
iteration = 0
ego_future_gen = scenario.get_ego_future_trajectory(iteration=iteration, time_horizon=8.0, num_samples=8)
ego_future = list(ego_future_gen)

print(f"Number of future states: {len(ego_future)}")
if len(ego_future) > 0:
    state = ego_future[0]
    print(f"\nFirst state type: {type(state)}")
    print(f"First state class name: {state.__class__.__name__}")
    print(f"\nFirst state attributes:")
    for attr in dir(state):
        if not attr.startswith('_'):
            print(f"  {attr}")
    
    print(f"\nChecking structure:")
    print(f"  hasattr(state, 'rear_axle'): {hasattr(state, 'rear_axle')}")
    if hasattr(state, 'rear_axle'):
        print(f"  state.rear_axle type: {type(state.rear_axle)}")
        print(f"  state.rear_axle: {state.rear_axle}")
        print(f"  state.rear_axle.x: {state.rear_axle.x}")
        print(f"  state.rear_axle.y: {state.rear_axle.y}")
        print(f"  state.rear_axle.heading: {state.rear_axle.heading}")
