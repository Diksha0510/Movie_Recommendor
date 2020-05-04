# Movie_Recommendor
ML based algorithm which recommends movies to users.

Prerequisities:

  Python libraries - sklearn, pandas
  
Method:

  Pearson correlation is used for comparaing similarity.
  Memory based Collaborative Filtering is used to recommend movies based on user-user or item-item correlations. In user-user   we try to find user's look alike based on previous ratings and in item-item we try to find movie's look alike.

Input:

  ratings.csv is the initial input file on which the model is trained. 
  
 
 Output:
 
  The predictions on the test data.
