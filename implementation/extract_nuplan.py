import os
import numpy as np
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import get_scenarios_from_log_file, GetScenariosFromDbFileParams

# Define paths
NUPLAN_DB_PATH = '/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-extracted/data/cache/mini/'
OUTPUT_DIR = 'implementation/cache/'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all database files
db_files = [os.path.join(NUPLAN_DB_PATH, f) for f in os.listdir(NUPLAN_DB_PATH) if f.endswith('.db')]

# Define parameters for scenario extraction
params_list = [
    GetScenariosFromDbFileParams(
        data_root=NUPLAN_DB_PATH,
        log_file_absolute_path=db_file,
        expand_scenarios=True,
        map_root='/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/maps/nuplan-maps-v1.0/',
        map_version='1.0',
        scenario_mapping=None,  # Assuming default mapping
        vehicle_parameters=None,  # Assuming default vehicle parameters
        filter_tokens=None,
        filter_types=['all'],
        filter_map_names=None,
        sensor_root=NUPLAN_DB_PATH,
        remove_invalid_goals=True
    ) for db_file in db_files
]

# Load scenarios
scenario_dicts = get_scenarios_from_log_file(params_list)

# Process and save each scenario
for scenario_dict in scenario_dicts:
    for scenario_type, scenarios in scenario_dict.items():
        for scenario in scenarios:
            # Extract data
            bev = scenario.get_bev_raster()
            ego_history = scenario.get_ego_history()
            gt_trajectory = scenario.get_ground_truth_trajectory()
            scenario_id = scenario.token

            # Save to .npz
            np.savez_compressed(
                os.path.join(OUTPUT_DIR, f'{scenario_id}.npz'),
                bev=bev,
                ego_history=ego_history,
                gt_trajectory=gt_trajectory,
                scenario_id=scenario_id
            )

print(f'Extraction complete. Files saved to {OUTPUT_DIR}')
