# vector_array_similarity

At SignalFrame, we had a notion of a signal environment as a collection of wifi and bluetooth signals that were available in an area, including a notion of how frequently a signal was present in an area.  These environments might share signals, and we we easily able to define similarity between two environments.

Vector Array Similarity built on top of that notion.  Different entities could be pass through several environments, with different amounts of time spent in each environment, and we would say that the entities were similar if their environments were similar.

However, several apparently obvious approaches to define similarity for entities in this way, ended up violating some core tenets of similarity.  Implementations of these approaches are included in similarity_test.py, with their failing tests included in SimilarityPropertiesTest.  The similarity function that we ultimately selected is in similarity.py.  Different rejected similarity functions from similarity_test.py can be substituted in at line 7 to expose which tests lead to its rejection.
