import warnings

from tqdm import tqdm

from wildfires.analysis.plotting import *
from wildfires.data.cube_aggregation import *
from wildfires.data.datasets import *

if __name__ == "__main__":
    warnings.filterwarnings("ignore", ".*Collapsing a non-contiguous coordinate.*")
    warnings.filterwarnings("ignore", ".*DEFAULT_SPHERICAL_EARTH_RADIUS*")

    thresholds = np.round(np.linspace(0.0, 0.95, 15), 3)
    mean_bas = []
    valid_counts = []

    for thres in tqdm(thresholds):
        obs_cube = CCI_BurnedArea_MERIS_4_1.get_obs_masked_dataset(
            "CCI MERIS BA", thres=thres
        )["CCI MERIS BA"]

        # Count the number of valid observations.
        valid_count = np.sum(~obs_cube.data.mask)

        # Calculate the mean burned area.
        mean_ba = obs_cube.collapsed(
            ("time", "latitude", "longitude"),
            iris.analysis.MEAN,
            weights=iris.analysis.cartography.area_weights(obs_cube),
        ).data

        valid_counts.append(valid_count)
        mean_bas.append(mean_ba)
