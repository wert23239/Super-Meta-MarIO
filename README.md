# Super Meta MarIO

## Using genetic algorithms and reinforcement learning combined to beat Mario levels.

## Preliminary Explanations

The whole concept of beating a game using code can be very confusing. This is because something like symbolic AI, Genetic Algorithms and Reinforcement Learning all looking very similar when showcasing. So, to help make this repo make more sense there a small helper guide to explain these very elaborate theories and what the problem some problems with them are.

### Symbolic AI (Hardcoding)

The easiest way to beat Mario without using a human is with traditional AI. This can be done in two ways. The first is very simple understand and how most Game AI works. All you simply have to do is hard code each input. This means everytime Mario is supposed to press right you just write a line of code that says go right. 

#### Problem (Not Actually AI)

This can also be called a TAS and has a very serious problem. There is no way to generalize this. It may be insanely accurate and work every single time. But, it can a follow a single path and is basically just a human beating the game through cheating. 

### Symbolic AI (Supervised Machine Learning)

Another way to beat Mario is through having a machine follow you. This is different from hard coding inputs because this is somewhat generalizable. What this involves is having a human play the game for a long period of time and then record all of its actions. Then a Neural Network looks at each position you were in and tries to guess what action the player did. This is how the OpenCV tutorial under this was able to make a self-driving car in GTAV.

#### Problem (Missing the point)

The problem isn't necessarily intuitive. For one it is very generalizable and there is still a neural network involved. However, it misses the point. If you need a human to start the initial process off then there's much better dataset that can create more useful products. Beating Mario using the methods below allows you to make a neural network learn without help. As this neural network may be able to may several Mario games it would be unable to do any other action without several more hours of human training. Not only this but the ending result could never be better than human since that's all its replicating. 

### Genetic Algorithms

This method is the first idea then would be useful if it could beat Mario. This would mean we could make intelligent artificial life with very simple instructions of what it was supposed to and a lot of time. A genetic algorithm is not a hard concept to understand. Essentially, all it does is make a random set of input to output mappings. After each set of mappings goes through the world they are all given a score or fitness. Then, the mappings who have the highest score or fitness carry on to the next generation. When the next generation starts all mappings are slightly altered or mutated. This means maybe if it used to see a block and jump it now ducks instead. Some of these mutations lower the score and some raise the score. Then the ones who now have the highest score carry on. You might even call it survival of the fittest. This is dumbing down a very complicated process but that's basically it.

#### Problem 1 (Infinte Time)

Genetic algorithms alone can technically beat anything. The one truth is that most will take so much time that it basically infinite. It's a lot like the monkey's on a typewriter or forming any combination using the number of pi. This is why some people just don't understand why this alone can't be the answer. Genetic algorithms are too slow. Any time a task starts getting too complicated it starts taking an exponential amount of time. 

#### Problem 2 (The Isolated Person Problem)

The most obvious problem even with infinite time is the concept of having a reactive machine. Imagine at every frame of the game, input, a machine spits back out the perfect buttons to press, output. This seems perfect and exactly in a perfect world what you would receive from a genetic algorithm. This could still not beat most games. Temporal information is very important. This is where the isolated person problem comes into play. Imagine a situation where all a character is walking in a completely empty room. How, does a human in this situation know whether to go left or right. The person simply used his past experience of which he started off as when the room wasn't empty. But, a machine that only maps inputs to outputs is unable to recall anything that happened even two seconds before this. Almost as if each organism is stuff in the Memento or Fifty First Dates movie.

#### Problem 3 (Stuck in a Valley)

The second only smaller scales has been solved. There is a genetic algorithm invented at my research called a Markov Brain. This organism has hidden states it can write to and therefore some ability to have memory. But, even with this problem solved you still can't beat Mario. This is because of a concept called Local Minimums. Imagine, a level where a user has to collect a key then enter a door. If the user doesn't collect the key then no matter what actions he does he will never do the truly best action. The algorithm may be able to spit out an organism that gets to the door extremely fast and believes it has the highest score possible. These local minimums are sometimes called Valleys are a huge problem with genetic algorithms. Often times an organism can do very well at first but will be unable to adapt to new situations and simply stop evolving. This concept of being very good at a small situation is called over-fitting.









### Tutuorials Used:

[MarIO "Genetic Algorithm" (Lua)](https://www.youtube.com/watch?v=qv6UVOQ0F44)

[Bizhawk "Emulator" (C#) ](http://tasvideos.org/BizHawk.html)

[OpenCv "Screen Capture" (Python)](https://www.youtube.com/watch?v=v07t_GEIQzI)

[DQN  "Reinforcement Learning"(Python Tenserflow)](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)

