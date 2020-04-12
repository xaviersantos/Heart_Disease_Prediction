import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def box_diagrams(data):
    data.plot(kind='box', subplots=True, layout=(3, 4), sharex=False, sharey=False)
    plt.savefig('report/images/box_whisker.pdf', bbox_inches='tight')
    plt.close()


def histogram(data):
    data.hist(bins=15, figsize=(10, 10), layout=(3, 2))
    plt.savefig('report/images/view_numerical.pdf', bbox_inches='tight')
    plt.close()


def view_categorical(data):
    fig, ax = plt.subplots(3, 3, figsize=(20, 10))

    for variable, subplot in zip(data.columns, ax.flatten()):
        sns.countplot(data[variable], ax=subplot, hue=data['Diagnosis'])
    plt.savefig('report/images/view_categorical.pdf', bbox_inches='tight')
    plt.close()


def percentage_general(data):
    ax = sns.countplot(data["Diagnosis"])
    num_count = data.Diagnosis.value_counts()

    # for showing the percentage
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 3, '{:1.2f}'.format(height / data.shape[0]), ha="center")

    plt.title("Percentage of heart problems in patients")
    plt.savefig('report/images/percentage_general.pdf', bbox_inches='tight')
    plt.close()
    print("\nPercentage of patience without heart problems: " + str(round(num_count[0] * 100 / 303, 2)))
    print("Percentage of patience with heart problems: " + str(round(num_count[1] * 100 / 303, 2)))


def age_distribution(data):
    sns.distplot(data['Age'])
    plt.savefig('report/images/age_distribution.pdf', bbox_inches='tight')
    plt.close()


def correlation(data):
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig('report/images/correlation_heatmap.pdf', bbox_inches='tight')
    plt.close()


def scatter_pairs(data):
    sns.pairplot(data, hue="Diagnosis", palette="husl")
    plt.savefig('report/images/correlation_scatter.pdf', bbox_inches='tight')
    plt.close()
