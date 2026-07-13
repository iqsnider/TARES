import csv
import matplotlib.pyplot as plt
import pandas as pd

def make_df(csv):
    """
    makes a pandas dataframe from a csv file
    """
    df = pd.read_csv(csv)

    return df


def acc_plot(df):
    """
    Compares the acceleration setpoints to IMU data for a given flight csv log
    """


