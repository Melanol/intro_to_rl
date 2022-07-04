This is an attempt to implement most algorithms and environments mentioned in the 1st 2 parts of the book "Reinforcement
Learning. An Introduction. Second Edition" by Sutton and Barto (http://incompleteideas.net/book/the-book-2nd.html). 
The idea is to make a collection of separate algos and envs a student can play with. No deep learning 
frameworks.

All algos and envs follow the same scheme (trying to copycat OpenAI's gym). All envs have a method step() returning 
(obs, reward, done). Also, the term "state" is replaced by "observation" (or "obs"). 
Envs usually have random agents inside (if makes sense). Run an environment to play it 
yourself, import it to teach an agent; run an algo to test it; run a Python figure file to replicate a figure. 

For envs with grids, the origin is in the top left corner.

Figure files mostly for replicating figures from the books, so algo files are main files.

Recommendation to self-learners: don't bother replicating everything in the book, it's not worth it. Read the book, 
understand it, play with algos, and then move on to https://github.com/DLR-RM/stable-baselines3.


# TODO: Use a switch for 2 players:
switch = 0
while True:
    if switch is 1:
        player1.act()
        switch = 1
    else:
        player2.act()
        switch = 0