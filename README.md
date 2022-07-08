WORK IN PROGRESS

This is an attempt to implement algorithms and environments mentioned in the 1st 2 parts of the book "Reinforcement
Learning. An Introduction. Second Edition" by Sutton and Barto (http://incompleteideas.net/book/the-book-2nd.html). 
The idea is to make a collection of separate algorithms and environments a student can play with. No deep learning 
frameworks.

All algorithms and environments follow the same scheme (trying to copycat OpenAI's gym). All environments have a method 
step() returning (obs, reward, done). Also, the term "state" is replaced by "observation" (or "obs"). Environments 
usually have random and human agents inside. Run an environment to play it yourself, import it into an algorithm 
to teach an agent, run a Python figure file to replicate a figure.

Figure files are mostly for replicating figures from the books, so algorithm files are main files.

Recommendation to self-learners: after you are done with the book, move on to 
https://github.com/DLR-RM/stable-baselines3.

This repository tries to not touch exercises.

I noticed 2 other worthy similar repositories: https://github.com/dennybritz/reinforcement-learning and 
https://github.com/ShangtongZhang/reinforcement-learning-an-introduction. With ShangtongZhang, I didn't like how
everything is glued together with the main idea to replicate images, and the code is pain to understand.
dennybritz's repository is OK, but I needed something in my CV anyway.
