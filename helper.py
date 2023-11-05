import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

def plot_differences(height_differences):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Height Differences Over Time')
    plt.xlabel('Time steps in this game')
    plt.ylabel('Sum of reward_height_diffs')
    plt.plot(range(len(height_differences)), height_differences)
    plt.ylim(ymin=0)
    plt.text(len(height_differences)-1, height_differences[-1], str(height_differences[-1]))
    plt.show(block=False)
    plt.pause(.1)