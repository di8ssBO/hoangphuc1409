import matplotlib.pyplot as plt
import seaborn as sns 

def SensorViz(df, feature_X, feature_y):
    plt.figure(figsize=(18,5))
    plt.scatter(df[feature_X],df[feature_y])
    plt.plot(df[feature_X],df[feature_y])
    plt.title(f'{feature_X} Vs {feature_y}', size = 20)
    plt.xlabel(feature_X, size = 20)
    plt.ylabel(feature_y, size = 20)
    plt.savefig(f'outputs/{feature_X} Vs {feature_y}.jpg')
    plt.grid()

def SensorVizWithPrediction(df, feature_X, feature_y, y_predict, savefig = False):
    plt.figure(figsize=(18,5))
    plt.scatter(df[feature_X],df[feature_y])
    plt.plot(df[feature_X],df[feature_y])
    plt.plot(df[feature_X], y_predict, color = 'red', linewidth=5)
    plt.title(f'{feature_X} Vs {feature_y}', size = 20)
    plt.xlabel(feature_X, size = 20)
    plt.ylabel(feature_y, size = 20)
    plt.savefig(f'outputs/Predicted {feature_X} Vs {feature_y}.jpg')
    plt.grid()