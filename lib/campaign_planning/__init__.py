"""
init file for CampaignPlanning
"""

# Let users know if they're missing any of our hard dependencies
hard_dependencies = ("numpy", "matplotlib", "pyomo", "pandas", "pygmo")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(f"{dependency}: {e}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(missing_dependencies)
    )
del hard_dependencies, dependency, missing_dependencies


from .misc import *
from .LET import *
from .vehicle_model import *
from .lunar_logistics import *
from .campaign_planner import *




