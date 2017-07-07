# Super Meta MarIO
## Using genetic algorithms and reinforcement learning  combined to beat mario levels.

#### Example

An example of a Local Minimum. If the score is how far right you go. An algorithm could evolve to simply always go right and miss the optimal path.

![Local Minimum](LocalMinimumBasic.png)


#### Example

This is same situation as described above. Without the abillity to retrain memory a genetic algorithm can't solve this problem.

![Isolated Person Problem](IsolatedPersonProblem.gif)

### Reinforcement Learning
 
This is the bleeding edge solution to beating Mario. To someone who hasn't worked with them and genetic alogrithms it is very hard to tell them apart. One of the major differnces if the learning process. Most reinforcement alogrithms try to do what supervised learning above does. But, instead of using a human to play the game and try to learn off of it. The alogrithm plays the game almost randomly and tracks of how well it did. It then creates its own testcases to learn from. Some stratedgies even include make a model simulated enviroment so it will always know how each action effects the enviroment.

#### Problem (Stiff)

There is almost no problems with this method. It has a lot of flexiabillity and comes in a variety of shapes and sizes. Even, the things that genetic alogrithm can't solve are pretty much solved here. For one, RNN (recurrent neural networks) allow for any reinforcement alogrithm to have some sense of memory. Besides that the exploration factor of most reinforcement stratdgies allow it to cross valleys very easily. The only problem is that no matter the algorithm the lack of an input/output table means that at each set of frames an action has to be picked out. But, as far as how many frames should the jump or right button be pressed is simply not an option. This results in a Mario that looks very stiff.

#### Problem (No Free Lunch Theory)

Another way to understand deep learning is trying to make an genetic alogritm that only evolves one orgranism. While, this analogy isn't perfect it does help visualize a major problem. In this method you are tring to make a single brain try to learn everything. This means if a single outlier level pops up like a water level all the weights in the network gets shifted over. This isn't a scalable solution to solve the whole game. 

#### Example

This repo contains a reinforcement learning algorithm playing Mario. This explains the similar problems discussed before. It also give negative score for going left which makes for a short sighted solution to certian problems even causing there learning curve to be convex.





#Mario AI github

## The Theory

The whole process of reinforcement learning really is very close the current best solution I know of. However, my idea has a slight twist to the concept. Instead of dealing with simple actions like a or b button. Have the network pick "complicated actions". This complicated action are actually rapidly changing micro genetic algorithms. In the same way Muscle Memroy works. Because the genetic alogrithms will be called at random times they will always start at slightly differnt places and make overfitting harder. This allows for a much more complex scalable system. However, since this no has ever done this there is almost no support for this and there is certain tricks to make this work.

## Prequisites


### Timeout Function

This idea comes from the Youtuber SethBling. In order to make a genetic algorithm evolve you have to let it play through the entire enviroment and when it is done it gets a score. This would normally mean in Mario you would have to let it play out one or three lives in its entirety. If at first the algorithm didn't even move this could take hours for a single generation. So instead SethBling decided if he knew the organism was not going anywhere and was at a stand still simply kill it off early. <b>This is concept is so important because it means an orgranism live can end before Mario actually dies.</b>

### Simple Continous Play
 
With the timeout function in place that means even after an orgranism times out most of the time Mario is still alive. What SethBling did was just use this for speed and had the next organism start from the beginning. But, instead for continous play you just allow the next orgranism start from where the last one gave up. This means again that not a single organsim has to try to beat the whole game

#### Example

This is a good example that showcases both a timeout and what non-continous and continous play looks like.
 

#### Problem (Impossible Fitness Function)

The biggest consequence of Continous Play is that you can start from anywhere. While, this prevents a lot of overfitting it also means a guy who's only good a running could be placed at a place where you need a very hard jump. Which causes a lot of randomness in the score or alogirthm which is called noise.

## The Solution Probalisitic Continous Play

To counteract this final problem. The organism can no longer be placed randomly. Instead there has to be an alogirthm that chooses where the genetic alogrithm are place. In other words a Machine Learning Alogrithm that helps a genetic alogirthm learn or Meta Machine Learning. This alogirithm of course ends of being a reinforcement learning algorithm. This is the goal of this repo.

 