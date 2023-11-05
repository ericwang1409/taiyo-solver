# taiyo-solver

## Inspiration
There is a game known as Suika Game, also known as the Watermelon Game, that has taken the world (and our dorm) by storm! We wanted to create an AI agent that could play this game at a superhuman level to defeat our friend, who holds the current dorm high score.

## What it does
To achieve our goals, we first created the entire original Taiyo Game (Planet Game) from scratch. The game can be played by anyone locally, and it will track and display your score to share with your friends! Second, we made another instance of the game that can be played by an autonomous Deep RL agent. Based on some of our experiments, and whatever reward or architectural ideas you come up with, it can either learn a lot from playing--or not so much!

## How we built it
We created the Taiyo Game from scratch using pygame, the pymunk physics engine, and DALLE-3 for assets. Then we made another version of Taiyo Game that has exit and insertion points for an autonomous agent to read the game state directly and input its desired action. We also created a Deep RL neural network using a custom implementation of the Deep Q-Learning algorithm with experience replay to train the agent on self-play of the game. This included creating a 'headless' version of the game that can simulate a game without displaying any graphics, allowing us to speed up the game tremendously. We also created extensive program versions for training on GPUs and TPUs, to varied success.

## Challenges we ran into
When we first made the game, we tried to make a custom physics engine from scratch, which was quite difficult. So, we had to refactor our game in terms of the pymunk physics engine that has compatibility with pygame. After we finished the game, there were a number of bugs to sort out, such as ball rendering, errant jumping, falling through the floor, etc.

When we started working with the agent and Deep RL, we found that specifying reward and training to better-than-random performance was difficult. A number of factors, such as left-stacking and the difficulty of learning complex behaviors such as managing a near-end game state, made it hard for the agent to perform well. We suspect that the required policy between game state and selected action would require significant training on a large neural network, since Taiyo is quite a deceptively complex game.

In order to train the model on a lot of games, we had to speed up the game, which causes pymunk physics calculations to require a bit more computational power. We had trouble getting the code to work on a GPU, which made training less reliable.

## Accomplishments that we're proud of
We are very proud of making the game from scratch, and as an easily-sharable and replayable experience, so that we can challenge our friends! Making the full game was more detailed and difficult than we expected, so being able to pull it off is really cool. We are also pleased that we could create the agent framework for inputting a training network and receiving board state followed by outputting agent action. We are also proud of many of our ideas for the reward signals.

## What's next for Taiyo Solver
We hope to spend time over the coming weeks (or at another Hackathon!) perfecting the reward signal and training on heavier computational power. We still need to destroy our friend's high score... with our computer science skills!
