# Project Planning

## 3D Bayes

Things left to do:
  - Code Bayesian Optimisation loop
  - Add sub-sampler for the virtual library 
  - Acquisition function 
  - Monte Carlo Sampler for the acquisition function
  - Distribution-level calibration "Distribution Calibration for Regression" by Son, Diethe, Kull & Flach (2019)
  - Read about Bayesian coresets ("Bayesian coresets" by Zhang, Khanna, Kyrillidis & Koyejo, 2021)

Follow up ideas:
  - Prove the Monte Carlo estimator is unbiased using a Dirilecht process for an empirical measure

Could it be the case that the paper on Riemannian Distributions help prove the previous statement by modelling the virtual library as a manifold and using a subsample distributed on the manifold to identify the Riemannian centre of mass.

## Amsterdam Project

We're currently on the planning stages. The project seems to be proceeding towards a generative approach.
  - Finish Floor's paper and list of questions

Given the paper, I've been having some thoughts. Does a re-framing from the flow based approach to a holistic dynamical system view help? If yes, could we model the distributions through Riemannian Gaussian Process and a target distribution by modelling the manifold's of the system. Do we have a stable/unstable/center manifolds? Does the current framework work with non-linear systems? If not, can we make them work by defining hyperbolic equilibrium points at one of the boundry conditions, namely the target distribution, in order to linearise them and identify the manifolds.
  - Papers to read:
    - "Non-stationary dynamical systems: shadowing theorem and some applications" by Mohtashamipour and Bahabadi (2023) about using an Anosov family to map using a sequence of diffeomorphisms along a sequence of compact Riemannian manifolds so that the tanget bundles split into expanding and contracting subspaces, with uniform bounds for the contraction and the expansion.
    - "Riemannian Gaussian Distributions on the Space of Symmetric Positive Definite Matrices" by Said, Bombrun, Berthoumieu & Manton (2016) about the connection between the definition of the title distribution and its connection with Riemannian centre of mass


## World Bank Project

Tasks to do:
  - Prepare heat-maps of the data
  - Do sensitivity analysis of the gating model
  - Check influence of the data while sequentially adding the factor
    - Save processed factors in new directory to simplify data loading

## Forecast Hub

This project is rather unimportant, I just have to finish the 

Parts left to be done:
  - Introduction
    - Rewrite introduction and draft the abstract
  - Background
    - Check with Fran how everything's progressing
  - Data retrieval
    - Save April datasets
      - Satellite images
      - Energy production
      - Environmental data
  - Methodology
    - Simplify to a normal ViT 
    - Use LSTM for the environmental data
    - Combine prediction as a MoE to simplify the approach and save time
  - Experiments and Results
    - Plot and interpret whatever comes out of the model
  - Discussion, Conclusion and Future Work
    - Discuss what other people have used, perhaps the recurrent visual transformer
    - Summarise what's been done and the main takeaways
    - Mention what could be done next and the met goals


## Machine Learning Theory

Final exam preparation:
  - Week 7
    - Final review
  - Week 8
    - Review notes
    - Do problem set 9
  - Week 9
    - Do notes
    - Do problem set 10
  - Week 10
    - Do notes
    - Do problem set 11
  - Week 11
    - Do Notes
    - Do problem set 12
  - Week 12
    - Do notes
    - Do problem set 13

