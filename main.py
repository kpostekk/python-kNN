from pandas import read_csv, DataFrame, concat
import matplotlib.pyplot as plt
from tkinter import Tk, Button


def load_dataframes():
    df: DataFrame = read_csv('data.csv', header=None)
    df['vector'] = [col for col in df.iloc[:, :-1].values]
    df.drop(list(range(df.columns.size - 2)), inplace=True, axis=1)
    df.columns = ['class', 'vector']

    train_df = DataFrame(columns=df.columns, data=None)
    test_df = DataFrame(columns=df.columns, data=None)

    # first 35 rows for training, last 15 for testing
    for class_name in df.groupby('class').groups:
        train_df = concat([train_df, df[df['class'] == class_name][:35]])
        test_df = concat([test_df, df[df['class'] == class_name][35:]])

    return train_df, test_df


def euclidean_distance(v_a, v_b):
    distance = 0.0
    for i in range(len(v_a)):
        distance += (v_a[i] - v_b[i]) ** 2
    return distance ** 0.5


def find_neighbors(train_df, test_row, num_neighbors):
    distances = list()
    for index, train_row in train_df.iterrows():
        dist = euclidean_distance(test_row['vector'], train_row['vector'])
        distances.append((train_row['class'], dist))
    distances.sort(key=lambda x: x[1])

    return distances[:num_neighbors]


def vote_neighbors(neighbors):
    class_votes = dict()

    for neighbor, _ in neighbors:
        if neighbor in class_votes:
            class_votes[neighbor] += 1
        else:
            class_votes[neighbor] = 1

    return max(class_votes.items(), key=lambda x: x[1])


def test_find_neighbors(k, train, test):
    passed = 0

    for _, test_row in test.iterrows():
        neighbours = find_neighbors(train, test_row, k)
        prediction, votes = vote_neighbors(neighbours)
        correct = prediction == test_row['class']

        if correct:
            passed += 1

    print(f'[{k=}] Passed: {passed}/{len(test)} ({passed / len(test) * 100:.2f}%)')
    return passed / len(test)


def init_gui():
    window = Tk()
    window.title('KNN')
    window.geometry('600x400')

    default_option_button = Button(command=default_config)
    default_option_button['text'] = 'Run with default options'
    default_option_button.pack()

    select_test_file_button = Button()
    select_test_file_button['text'] = 'Select test file'
    select_test_file_button.pack()

    insert_own_input_button = Button()
    insert_own_input_button['text'] = 'Insert own input'
    insert_own_input_button.pack()

    window.mainloop()


def default_config():
    train_df, test_df = load_dataframes()
    dfm = DataFrame(
        columns=['k', 'accuracy'],
        data=[[k, test_find_neighbors(k, train_df, test_df)] for k in range(1, 45)]
    )
    dfm.plot.line(x='k', y='accuracy')
    print(dfm)
    plt.show()


if __name__ == '__main__':
    init_gui()
