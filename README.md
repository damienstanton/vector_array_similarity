# Vector Array Similarity 

Often an object can be represented as an unordered set of vectors/embeddings.
In particular we are interested in comparing objects that have multiple embeddings
in the same vector space. In other words all vectors share the same dimensions.  

These representations naturally arise in time-series analysis where the behaviour
changes over time within the same vector space. E.g. a user in social network 
creates different personas over time.

For example:
```
user_1 := [v11, v12, v13]
user_2 := [v21, v22]
```
where _v_ denotes a vector that defines some behaviour in a given time window. 
What is the similarity between the two users? Note that we do expect that each user
may have different number of vectors due to different activity levels and amount of 
time spent.
 
This package implements a similarity function that takes into account how important 
individual vectors are to a given user, and also accounts the different array lengths.
In particular, absence of vectors is interpreted through the Closed-World Assumption
as lack of activity. 
