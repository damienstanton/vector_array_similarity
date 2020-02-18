# Vector Array Similarity 

## Problem Setting 

Often an object can be represented as an unordered set of vectors/embeddings.
In particular we are interested in comparing objects that have multiple embeddings
in the same vector space. In other words all vectors share the same dimensions.  

These representations naturally arise in time-series analysis where the behaviour
changes over time within the same vector space. E.g. a user in social network 
creates different personas over time.

For example:
```
user1 := [v11, v12, v13, v14]
user2 := [v21, v22, v15]
```
where _v[UserId][TimeWindow]_ denotes a vector that defines some behaviour in a given time window. 
What is the similarity between the two users? Note that we do expect that each user
may have different number of vectors due to different activity levels and amount of 
time spent. In this case, user2 has no activity during windows 3, 4; user1 has no activity during
window 5.

## A Solution

This package implements an _opiniated_ similarity function that takes into account how important 
individual vectors are to a given user, and also accounts for the different array lengths (i.e. 
available time windows).In particular, absence of vectors is interpreted through the Closed-World 
Assumption as lack of activity. 
