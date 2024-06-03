#  Choice between CNN and RNN if new: None
model = "RNN"
approach = "regression"
random_seed = 44
random_state_train = 44
sequence = 12

#######################################################################################################################

#  Features
# features = ["y", "number_of_labs", "number_of_employees", "number_of_pir_doctors", "number_of_stuff", "number_of_doctors",
#             "number_of_persons_in_hotels", "number_of_travels_bus_outside", "number_of_travels_minibus_outside", "covid", "month",
#             "number_of_travels_car_outside", "number_of_person_bus_outside", "number_of_person_minibus_outside", "number_of_person_car_outside"]
features = ["y", "number_of_active_beds", "number_of_person_bus_outside", "number_of_travels_bus_outside", "number_of_beds",
            "number_of_labs", "number_of_employees", "number_of_doctors",  "markaze_behdasht", "covid", "month"]

# features = ["y", "number_of_person_car_outside", "number_of_travels_car_outside", "number_of_person_bus_outside",
#             "number_of_travels_minibus_inside", "number_of_person_bus_inside", "number_of_travels_bus_outside",
#             "number_of_travels_minibus_outside", "covid", "number_of_active_beds", "number_of_beds", "markaze_behdasht"]
#######################################################################################################################

#  Target
target = ["y"]

#######################################################################################################################

#  Model identification
model_name = "Health"
model_architecture = "RNN"
model_version = "V0"

#######################################################################################################################

columns = features + target
