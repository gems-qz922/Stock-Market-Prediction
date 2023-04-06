import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf


class stock:
    def __init__(self, code):
        '''code: code of a stock'''
        self.code = code

    def get_data(self):
        '''The function to get historic price information of a stock.
        code: string of the 6 digit number of a stock'''
        info = yf.ticker('tsla')
        print(info)
stock(1)