# RAMP starting kit on Book Recommender System

# Instructions
Authors: Youssef Brachmi, Théo Dullin, Omar El Mellouki, Lokmen Eltarr, Maxime Berillon

In their article *Beyond Books: The Extended Academic Benefits of Library Use for First-Year College Students*  Soria et al. described how the use of library and most importantly book can have a positive impact on the academic outcomes of college students. Therefore it is crucial to advise them well.

In another study called *A book reading intervention with preschool children who have limited vocabularies: the benefits of regular reading and dialogic reading*, Hargrave et al. focused on a younger generation and conclude in a similar way. They studied the influence of book reading to children who had poor vocabulary skills. These book reading interventions were very conclusive : the children exposes to these sessions rapidly gained vocabulary. This further stresses our point that book are vital for personal developement.

The challenge that is proposed here is to build a book recommander system. Based on information about a list of books and the reviews of some users, the aim is to predict a list of book that a user is the most likely to like i,e, that he would review with a high rating.

#### Set up

Open a terminal and

1. install the `ramp-workflow` library (if not already done)
  ```
  $ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
  ```
  
2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)

#### Local notebook

Get started on this RAMP with the [dedicated notebook](https://github.com/maximeberillon/Recommandation_System_Books/blob/main/book_reco_startingkit.ipynb).

To test the starting-kit, run on terminal:

```
ramp_test_submission --submission starting_kit
```
or for a quick version of the test, just run:

```
ramp_test_submission --submission starting_kit --quick-test
```
