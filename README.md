WORK IN PROGRESS

This is an attempt to implement most algorithms and environments mentioned in the 1st 2 parts of the book "Reinforcement
Learning. An Introduction. Second Edition" by Sutton and Barto (http://incompleteideas.net/book/the-book-2nd.html). 
The idea is to make a collection of separate algos and envs a student can play with. No deep learning 
frameworks.

All algos and envs follow the same scheme (trying to copycat OpenAI's gym). All envs have a method step() returning 
(obs, reward, done). Also, the term "state" is replaced by "observation" (or "obs"). Envs usually have random and 
human agents inside. Run an environment to play it yourself, import it into an algo to teach an agent, run a Python 
figure file to replicate a figure.

For envs with grids, the origin is in the top left corner.

Figure files mostly for replicating figures from the books, so algo files are main files.

Recommendation to self-learners: after you are done with the book, move on to 
https://github.com/DLR-RM/stable-baselines3.

This repo tries to not touch exercises.

I noticed 2 other worthy similar repos: https://github.com/dennybritz/reinforcement-learning and 
https://github.com/ShangtongZhang/reinforcement-learning-an-introduction. With ShangtongZhang I didn't like how
everything was glued together with the main idea to replicate images, and the code was pain to understand.
dennybritz's repo was OK, but I needed something in my CV anyway.

TODO: Normalize hyper names (gamma or discount)
