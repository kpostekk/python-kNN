import click
from matplotlib import pyplot as plt
from pandas import DataFrame

from knn import test_find_neighbors, load_dataframes, load_discrete_df, find_neighbors, vote_neighbors


@click.group()
def cli():
    pass


@cli.command()
@click.option('--k-min', default=1, type=int)
@click.option('--k-max', default=45, type=int)
def plot(k_min, k_max):
    train_df, test_df = load_dataframes()
    dfm = DataFrame(
        columns=['k', 'accuracy'],
        data=[[k, test_find_neighbors(k, train_df, test_df)] for k in range(k_min, k_max)]
    )
    dfm.plot.line(x='k', y='accuracy')
    print(dfm)
    plt.show()


@cli.command()
@click.option('-k', default=3, type=int)
def default(k):
    train_df, test_df = load_dataframes()
    ac = test_find_neighbors(k, train_df, test_df)
    print(f'Accuracy: {ac * 100:.2f}%')


@cli.command()
@click.option('-k', default=3, type=int)
@click.option('--test-file', type=str)
def custom(k, test_file):
    train_df, _ = load_dataframes()
    test_df = load_discrete_df(test_file)
    ac = test_find_neighbors(k, train_df, test_df)
    print(f'Accuracy: {ac * 100:.2f}%')


@cli.command()
@click.option('-k', default=3, type=int)
@click.option('--input-v', type=str)
def predict(k, input_v):
    train_df, _ = load_dataframes()
    neighbors = find_neighbors(train_df, {'vector': eval(input_v)}, k)
    prediction, votes = vote_neighbors(neighbors)
    print(f'Prediction: {prediction} ({votes} votes)')


if __name__ == '__main__':
    cli()
