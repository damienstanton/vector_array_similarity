# Vector Array Similarity 

## Problem Setting 

Often an object can be represented as a set of vectors/embeddings.
In particular, we are interested in comparing objects that have multiple embeddings
in the same vector space. In other words all vectors share the same dimensions.  

These representations naturally arise in time-series analysis where the behaviour
changes over time within the same vector space. E.g. a user in social network 
creates different personas over time.

For example:
```
user1 := [v11, v12, v13, v14]
user2 := [v21, v22, v25]
```
where _v[UserId][TimeWindow]_ denotes a vector that defines some behaviour in a given time window. 
What is the similarity between the two users? Note that we do expect that each user
may have different number of vectors due to different activity levels and amount of 
time spent. In this case, user2 has no activity during windows 3, 4, while user1 has no activity during
window 5.

## Given Solution

This package implements an _opiniated_ similarity function that takes into account how important 
individual vectors are to a given user, and accounts for the different array lengths (i.e. 
available time windows). In particular, absence of vectors is interpreted, through the Closed-World 
Assumption, as lack of activity.

This similarity function does not care about the time ordering of vectors, just their presence. Thus
two users are highly similar if their vectors are similar regardless of their time position. However,
if user1 has only one vector and it happens to be highly similar to one of user2's vectors, but user2 has
additional vectors, then their similarity will be diminished. In other words, lack of activity influences
the similarity measure.
