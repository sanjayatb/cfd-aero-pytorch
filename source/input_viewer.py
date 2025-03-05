import argparse

from source.data.data_visualize import InputDataViewer
from source.data.enums import CFDDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run Experiment")
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        choices=[e.value for e in CFDDataset],
    )
    parser.add_argument(
        "--file-name",
        type=str
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=100000
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_set = None
    if args.dataset_name == CFDDataset.AHMED_ML.value:
        data_set = "ahmed_ml"
    elif args.dataset_name == CFDDataset.WINDSOR_ML.value:
        data_set = "windsor_ml"
    elif args.dataset_name == CFDDataset.DRIVAER_ML.value:
        data_set = "drivaer_ml"
    else:
        raise NotImplementedError(f"{args.dataset_name} not implemented.")

    viewer = InputDataViewer()

    # viewer.show_targets("../inputs/{data_set}/targets.csv")

    viewer.view_stl(f"../inputs/{data_set}/stl/")
    viewer.visualize_point_cloud(f"../inputs/{data_set}/stl/{args.file_name}", args.num_points)
