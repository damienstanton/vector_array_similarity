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

## A Solution

This package implements an _opiniated_ similarity function that takes into account how important 
individual vectors are to a given user, and accounts for the different array lengths (i.e. 
available time windows). This similarity function does not care about the time ordering of vectors, just their presence. Thus
two users are highly similar if their vectors are similar regardless of their time position. 

This solution also defends against cases where arrays are of different lengths, yet most of the elements are similar.
For example, if user1 has only one vector and it happens to be highly similar to one of user2's vectors, but user2 has
additional vectors, then their similarity will be diminished. In other words, lack of activity influences
the similarity measure.

### Algorithm Sketch

```
INPUT ::= {
		matrix M: of pair-wise cosine distances between vectors
		weights_a: vector of assigned weights to each vector in array a
		weights_b: vecotr of assigned weights to each vector in array b 
}

1. build W such that w(i,j) = min(weights_a(i), weights_b(j))

2. S <- M o W // Hadamard product

3. score = MAX(SUM(i to n) max(S(i,.)), SUM(j to m) MAX(S(.,j)))

4. decay = MAX(0norm(MAX(i to n) M(i,.)), 0norm(MAX(j to m) M(.,j)))
4.1 	def fn 0norm(vector) ::= (len_non_zero(vector) + 1)/(len(vector) +1)

5. return score * decay
```

## Distribution
This software is distrubted under the MIT license. It will not be actively developed further.
