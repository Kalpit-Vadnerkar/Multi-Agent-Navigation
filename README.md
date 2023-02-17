# Multi-Agent-Navigation
Approaches:
- Social Force Model:

	The Social Force Model was relatively easy to implement. The ksi value for this approach was set to 0.5. The problem with this approach is that as the number of agents increace, so does the crowding. Sure, it figures out the way in the end but it feels like it could have been done better. 
- Predictive Time-To-Collision (TTC) Forces:

	The TTC approach almost solves the crowding problem but not completly. The agents  converge to the center and then move in a circular fashion, like cars at a traffic circle. The path that the agents took was pretty close to each other. Implementing with the eplsilon = 0.2 the agents took a more safe path. 
- Sampling Based Velocity:

	I was impressed with the solution acquired through this approach as it seems like the agents have some information about the environment even though they dont. Once i figured out the TTC approach, it was easy to code this one. I sampled 100 candidate velocities as it gave me an output i was satisfied with. Sampling 1000 candidate velocities definitely makes the motion of the agents smooth but it was computationally expensive for the machine i am working with.
