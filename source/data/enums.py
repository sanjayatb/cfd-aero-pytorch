from enum import Enum


class CFDDataset(Enum):
    AHMED_ML = "AhmedML"
    WINDSOR_ML = "WindsorML"
    DRIVAER_ML = "DrivAerML"
    DRIVAER_NET = "DrivAerNet"


class ModelArchitecture(Enum):
    POINT_NET = "PointNet"
    GNN = "GNN"
    FNO = "FNO"
    GINO = "GINO"
