
import pickle
import json
import pandas as pd
import numpy as np

class MedicalInsurance():
    def __init__(self, age, sex, bmi, children, smoker, region):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = "region_" + region

    def load_model(self):
        with open("Linear_regression.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open("project_data.json", "r") as f:
            self.json_data = json.load(f)#instance variable 

    def get_predicted_price(self):

        self.load_model()        # Calling load_model method to get model and json_data

        region_index = self.json_data['columns'].index(self.region)

        array = np.zeros(len(self.json_data['columns']))

        array[0] = self.age
        array[1] = self.json_data['sex'][self.sex]
        array[2] = self.bmi
        array[3] = self.children
        array[4] = self.json_data['smoker'][self.smoker]
        array[region_index] = 1

        print("Test Array -->\n",array)
        predicted_charges = self.model.predict([array])[0]
        print("predicted_charges",predicted_charges)
        return np.around(predicted_charges, 2)


if __name__ == "__main__":
    age = 67
    sex = "male"
    bmi = 27.9
    children = 3
    smoker = "yes"
    region = "southeast"

    med_ins = MedicalInsurance(age, sex, bmi, children, smoker, region)
    charges = med_ins.get_predicted_price()
    print()
    print(f"Predicted Charges for Medical Insurance is {charges}/- Rs. Only")