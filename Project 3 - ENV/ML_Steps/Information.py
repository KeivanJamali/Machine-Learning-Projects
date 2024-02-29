#  Choice between CNN and RNN if new: None
model = "RNN"
approach = "regression"
random_seed = 42

#######################################################################################################################

#  Features
features = ["precipitation_mean", "Wind speed", "SoilMoi0_10cm_inst", "SoilTMP0_10cm_inst", "area_Water"]

#######################################################################################################################

#  Target
target = ["Optical_Depth_055"]

#######################################################################################################################

#  For RNN type
#  Sequence or None
sequence = 4
number_layer = 4

#######################################################################################################################

#  Model identification
model_name = "AOV"
model_architecture = "RNN"
model_version = "V0"

#######################################################################################################################

columns = features + target
